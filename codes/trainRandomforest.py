#!/usr/bin/env python3
"""
train_randomforest_fixed.py

Refined, robust, and fast RandomForest training pipeline for your Tox21-style data.
Saves per-assay models into ./modelrandomforest and writes a training_summary.csv.

Features:
 - Uses 2048-bit Morgan fingerprints (same as your prediction script expects)
 - Robust label detection + sensible binarization rules
 - Handles messy / multiple label files per assay
 - Skips assays with too few samples or single-class labels
 - Uses class_weight='balanced' to reduce bias from imbalanced classes
 - Fast fingerprint generation (uses swifter if available, falls back to pandas apply)
 - Saves per-assay models as: ./modelrandomforest/{ASSAY}_model.pkl
"""

import os
import glob
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score

# -------------------------
# Config
# -------------------------
SMILES_PATH = "/home/hegde_go/Documents/AIML-LAB-EL/tox21_10k_challenge_score.smiles"
LABEL_ROOT  = "/home/hegde_go/Documents/AIML-LAB-EL/tox21-challenge"
OUT_DIR     = "./modelrandomforest"
MIN_SAMPLES = 50            # skip assays with fewer matched samples
TEST_SIZE   = 0.20
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
RDLogger.DisableLog('rdApp.*')  # silence RDKit logs

# try to import swifter for faster apply; if missing, we fall back
try:
    import swifter  # noqa: F401
    SWIFTER_AVAILABLE = True
except Exception:
    SWIFTER_AVAILABLE = False

# -------------------------
# Feature builder (2048-bit Morgan FP only)
# -------------------------
def featurize_to_2048(smiles):
    """Return numpy array (2048,) of bits (0/1) or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.array(fp, dtype=np.uint8)  # small memory
        return arr
    except Exception:
        return None

# wrapper for pandas apply (uses swifter if available)
def featurize_series(series):
    if SWIFTER_AVAILABLE:
        return series.swifter.apply(featurize_to_2048)
    else:
        # fallback: use list comprehension for speed + progress via tqdm
        res = []
        for smi in tqdm(series, desc="Featurizing", ncols=80, unit="mol"):
            res.append(featurize_to_2048(smi))
        return pd.Series(res, index=series.index)

# -------------------------
# Label cleaning & binarization rules
# -------------------------
def detect_id_and_activity_cols(df):
    """Return (id_col, act_col) or (None, None)."""
    cols = {c.lower(): c for c in df.columns}
    id_candidates = ["sample id", "sample_id", "sampleid", "id", "molecule id", "compound id"]
    act_candidates = ["activity", "activity_score", "score", "activityscore", "result"]

    id_col = next((cols[c] for c in id_candidates if c in cols), None)
    act_col = next((cols[c] for c in act_candidates if c in cols), None)
    return id_col, act_col

def binarize_activity(series):
    """
    Robust binarization:
     - if values appear to be categorical {-1,0,1} or 0/1: treat >=1 as active
     - else if continuous and range > 1 => assume score in [0,1] or percentages => use >0.5
     - else default: round and treat 1 as active
    Returns int Series (0/1) and original numeric series.
    """
    s_num = pd.to_numeric(series, errors="coerce")
    s_num = s_num.dropna()
    if s_num.empty:
        return None

    mx = s_num.max()
    mn = s_num.min()
    # obvious categorical {-1,0,1} or {0,1}:
    if set(np.unique(s_num)).issubset({-1, 0, 1}) or mx <= 1.1:
        # treat >=1 as active (cover both 1 or >0.5 cases where values are 0/1 or -1/0/1)
        b = (s_num >= 1).astype(int)
        # if no ones and there are values like 0.6/0.8, then treat >0.5 as active
        if b.sum() == 0 and mx > 0.5:
            b = (s_num > 0.5).astype(int)
    else:
        # continuous values larger than 1 (maybe percentages or raw scores) -> threshold 0.5
        b = (s_num > 0.5).astype(int)

    # map back to original index
    b = b.reindex(series.index).fillna(0).astype(int)
    return b, s_num

# -------------------------
# Utility: read label files robustly
# -------------------------
def read_and_concat_label_files(path_list):
    frames = []
    for p in path_list:
        try:
            df = pd.read_csv(p, sep="\t", engine="python", dtype=str)
            # strip column names / whitespace
            df.columns = [c.strip() for c in df.columns]
            frames.append(df)
        except Exception as e:
            print(f"  ⚠ Warning: failed to read {p}: {e}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)

# -------------------------
# Main per-assay processing
# -------------------------
summary = []

# Load SMILES master file
print(f"Loading SMILES from: {SMILES_PATH}")
smiles_df = pd.read_csv(SMILES_PATH, sep=r"\s+", names=["SMILES", "ID"],
                        comment="#", engine="python", dtype=str, keep_default_na=False)
smiles_df["SMILES"] = smiles_df["SMILES"].astype(str).str.strip()
smiles_df["ID"] = smiles_df["ID"].astype(str).str.strip()
# dedupe by ID (keep first)
smiles_df = smiles_df.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)
print(f" Total SMILES entries: {len(smiles_df)}")

assay_folders = sorted([d for d in os.listdir(LABEL_ROOT) if os.path.isdir(os.path.join(LABEL_ROOT, d))])
print("Assays found:", assay_folders)

for assay in assay_folders:
    print("\n" + "-"*60)
    print(f"Processing assay: {assay}")
    label_paths = glob.glob(os.path.join(LABEL_ROOT, assay, "*.txt"))
    if not label_paths:
        print("  ⚠️ no label files found — skipping")
        summary.append((assay, 0, 0, "no_label_files", None, None))
        continue

    labels_df = read_and_concat_label_files(label_paths)
    if labels_df is None:
        summary.append((assay, 0, 0, "failed_read", None, None))
        continue

    id_col, act_col = detect_id_and_activity_cols(labels_df)
    if id_col is None or act_col is None:
        print(f"  ⚠️ Could not detect ID/Activity columns. Found cols: {list(labels_df.columns)} — skipping")
        summary.append((assay, 0, 0, "missing_cols", None, None))
        continue

    # drop NA ids and activities
    labels_df = labels_df.dropna(subset=[id_col, act_col])
    labels_df[id_col] = labels_df[id_col].astype(str).str.strip()

    # binarize activity robustly
    try:
        binarized_result = binarize_activity(labels_df[act_col])
        if binarized_result is None:
            print("  ⚠️ could not binarize activity — skipping")
            summary.append((assay, 0, 0, "binarize_failed", None, None))
            continue
        binary_activity_series, _ = binarized_result
    except Exception as e:
        print(f"  ⚠️ binarize step failed: {e} — skipping")
        summary.append((assay, 0, 0, "binarize_exception", None, None))
        continue

    # assemble cleaned labels frame
    labels_clean = pd.DataFrame({
        "ID": labels_df[id_col].values,
        "Activity": binary_activity_series.values
    })
    # aggregate by ID (take max activity per ID)
    labels_agg = labels_clean.groupby("ID", as_index=False)["Activity"].max()

    # merge with SMILES master
    merged = smiles_df.merge(labels_agg, left_on="ID", right_on="ID", how="inner")
    print(f"  Rows after merge: {len(merged)}")
    if len(merged) < MIN_SAMPLES:
        print(f"  ⚠️ Too few matched samples (<{MIN_SAMPLES}) — skipping")
        summary.append((assay, len(merged), int(merged["Activity"].sum()), "too_small", None, None))
        continue

    # drop duplicates, invalid SMILES
    merged = merged.drop_duplicates(subset=["ID", "SMILES"])
    # featurize
    print(f"  Featurizing {len(merged)} molecules...")
    merged["FP"] = featurize_series(merged["SMILES"])

    # remove rows where FP is None
    merged = merged[merged["FP"].notna()].reset_index(drop=True)
    print(f"  Valid samples after fingerprinting: {len(merged)}")
    if len(merged) < MIN_SAMPLES:
        print(f"  ⚠️ Too few valid fingerprints (<{MIN_SAMPLES}) — skipping")
        summary.append((assay, len(merged), int(merged["Activity"].sum()), "too_few_fp", None, None))
        continue

    X = np.vstack(merged["FP"].values).astype(np.float32)  # shape (N, 2048)
    y = merged["Activity"].astype(int).values

    # check class balance
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(f"  Activity distribution: {class_counts}")
    if len(unique) < 2 or min(counts) < 2:
        print("  ⚠️ Single-class or too few positives — skipping")
        summary.append((assay, len(y), int(np.sum(y)), "single_class", None, None))
        continue

    # split (stratified if possible)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                            random_state=RANDOM_SEED, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                            random_state=RANDOM_SEED)

    # compute class weight ratio for logging
    pos_train = int((y_train == 1).sum())
    neg_train = int((y_train == 0).sum())
    print(f"  Training set: pos={pos_train}, neg={neg_train}")

    # build RF with balanced class weight
    clf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED
    )

    print("  Training RandomForest...")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    auc = None
    if y_prob is not None and len(np.unique(y_test)) == 2:
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            auc = None

    print("  Results:")
    print(f"    Accuracy     : {acc:.4f}")
    print(f"    Balanced ACC : {bacc:.4f}")
    print(f"    ROC-AUC      : {auc if auc is not None else 'N/A'}")

    # save model
    out_path = os.path.join(OUT_DIR, f"{assay}_model.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(clf, fh)
    print(f"  Saved model: {out_path}")

    summary.append((assay, len(y), int(np.sum(y)), "trained", round(acc, 4), round(bacc, 4), (round(auc, 4) if auc is not None else None)))

# -------------------------
# Save summary CSV
# -------------------------
summary_rows = []
for s in summary:
    if s[3] == "trained":
        summary_rows.append({
            "Assay": s[0], "Samples": s[1], "Positives": s[2],
            "Status": s[3], "Accuracy": s[4], "Balanced_ACC": s[5], "ROC_AUC": s[6]
        })
    else:
        summary_rows.append({
            "Assay": s[0], "Samples": s[1], "Positives": s[2],
            "Status": s[3], "Accuracy": None, "Balanced_ACC": None, "ROC_AUC": None
        })

summary_df = pd.DataFrame(summary_rows)
csv_path = os.path.join(OUT_DIR, "training_summary.csv")
summary_df.to_csv(csv_path, index=False)
print("\nTraining complete. Summary saved to:", csv_path)
