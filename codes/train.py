#!/usr/bin/env python3
"""
Refined training pipeline for Tox21-style SMILES + assay labels.

- Expects a SMILES file like your uploaded:
    #SMILES\tSample ID
    OC(=O)...\tNCGC00261900-01
  (header commented with '#', tab or whitespace separated)

- Expects label folders under LABEL_ROOT, each containing *.txt label files
  similar to: "Sample ID\tScore\tActivity"

- Produces per-assay models saved as: models/<ASSAY>.pkl
- Produces a summary CSV at: models/training_summary.csv

Run:
    python train_refined.py
"""
import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef

# xgboost is used (fast & robust). Install with: conda install -c conda-forge xgboost  OR pip install xgboost
from xgboost import XGBClassifier

# -----------------------------
# CONFIG
# -----------------------------
SMILES_FILE = "/home/hegde_go/Documents/AIML-LAB-EL/tox21_10k_challenge_score.smiles"
LABEL_ROOT  = "/home/hegde_go/Documents/AIML-LAB-EL/tox21-challenge"
OUT_DIR     = "./models"
MIN_SAMPLES = 50           # skip assays with fewer samples
TEST_SIZE   = 0.20
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
RDLogger.DisableLog('rdApp.*')   # silence RDKit

# -----------------------------
# FEATURIZATION (match training features)
# -----------------------------
def featurize(smile):
    """Return a 2054-d numpy array: 2048-bit Morgan FP + 6 descriptors"""
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_arr = np.array(fp, dtype=np.float32)

        desc = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol)
        ], dtype=np.float32)

        return np.concatenate([fp_arr, desc])
    except Exception:
        return None

# -----------------------------
# MODEL FACTORY
# -----------------------------
def make_model(scale_pos_weight=1.0):
    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=float(scale_pos_weight),
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

# -----------------------------
# LOAD SMILES
# -----------------------------
print("Loading SMILES:", SMILES_FILE)
smiles_df = pd.read_csv(
    SMILES_FILE,
    sep=r"\s+",
    names=["SMILES", "ID"],
    comment="#",
    engine="python",
    dtype=str,
    keep_default_na=False
)

# sanitize strings
smiles_df["SMILES"] = smiles_df["SMILES"].astype(str).str.strip()
smiles_df["ID"]     = smiles_df["ID"].astype(str).str.strip()

# deduplicate by ID (keep first)
smiles_df = smiles_df.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)
print(f" Loaded {len(smiles_df)} unique SMILES entries.")

# -----------------------------
# FIND ASSAY FOLDERS
# -----------------------------
assay_folders = sorted([d for d in os.listdir(LABEL_ROOT) if os.path.isdir(os.path.join(LABEL_ROOT, d))])
print("Assays found:", assay_folders)

summary_rows = []

# -----------------------------
# PROCESS EACH ASSAY
# -----------------------------
for assay in assay_folders:
    print("\n" + "-"*60)
    print(f"Processing assay: {assay}")

    label_paths = glob.glob(os.path.join(LABEL_ROOT, assay, "*.txt"))
    if not label_paths:
        print(" No label files found; skipping.")
        continue

    # read & concat label files robustly
    label_frames = []
    for lp in label_paths:
        try:
            df = pd.read_csv(lp, sep="\t", engine="python", dtype=str)
            df.columns = [c.strip() for c in df.columns]
            label_frames.append(df)
        except Exception as e:
            print(f"  Warning: failed reading {lp}: {e}")

    if not label_frames:
        print(" No readable label files; skipping.")
        continue

    labels = pd.concat(label_frames, ignore_index=True, sort=False)

    # detect id column
    colmap = {c.lower(): c for c in labels.columns}
    id_col_candidates = ["sample id", "sample_id", "id", "molecule id", "compound id"]
    activity_candidates = ["activity", "score", "activity_score"]

    id_col = next((colmap[k] for k in id_col_candidates if k in colmap), None)
    act_col = next((colmap[k] for k in activity_candidates if k in colmap), None)

    if id_col is None or act_col is None:
        print(f" Could not detect ID/Activity columns (found cols: {list(labels.columns)}) — skipping.")
        continue

    # Keep necessary columns and clean
    labels = labels[[id_col, act_col]].dropna(how="any")
    labels[id_col] = labels[id_col].astype(str).str.strip()
    # convert activity to numeric
    labels[act_col] = pd.to_numeric(labels[act_col], errors="coerce")
    labels = labels.dropna(subset=[act_col])

    # binarize: if score looks continuous (>1) treat threshold >0.5 else round
    if labels[act_col].max() > 1.1:
        labels["Activity"] = (labels[act_col] > 0.5).astype(int)
    else:
        labels["Activity"] = labels[act_col].round().astype(int)

    # aggregate by ID (take max activity per ID)
    labels = labels.groupby(id_col, as_index=False)["Activity"].max()
    labels.rename(columns={id_col: "ID"}, inplace=True)

    # merge with SMILES
    merged = smiles_df.merge(labels, left_on="ID", right_on="ID", how="inner")
    print(f" Samples after merge: {len(merged)}")

    if len(merged) < MIN_SAMPLES:
        print(f" Too few samples (<{MIN_SAMPLES}) — skipping.")
        summary_rows.append((assay, len(merged), np.sum(merged.get("Activity", 0)), "skipped_too_small"))
        continue

    # featurize
    print(" Featurizing SMILES...")
    feats = []
    acts  = []
    for smi, act in tqdm(zip(merged["SMILES"], merged["Activity"]), total=len(merged), ncols=80, unit="mol"):
        v = featurize(smi)
        if v is None:
            continue
        feats.append(v)
        acts.append(int(act))

    if len(feats) == 0:
        print(" No valid featurized molecules — skipping.")
        summary_rows.append((assay, 0, 0, "skipped_no_valid_smiles"))
        continue

    X = np.vstack(feats)
    y = np.array(acts, dtype=int)
    print(f" Final dataset shape: X={X.shape}, y={y.shape}")

    # check class balance
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(" Activity distribution:", class_counts)

    if len(unique) < 2 or min(counts) < 2:
        print(" Insufficient class variety — skipping.")
        summary_rows.append((assay, len(y), int(np.sum(y)), "skipped_single_class"))
        continue

    # split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
    except ValueError:
        # fallback without stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    if pos == 0 or neg == 0:
        print(" Train set became single-class after split — skipping.")
        summary_rows.append((assay, len(y), int(np.sum(y)), "skipped_after_split"))
        continue

    scale_weight = neg / (pos + 1e-6)
    model = make_model(scale_weight)

    print(f" Training model (pos={pos}, neg={neg}, scale_pos_weight={scale_weight:.2f}) ...")
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f" Model training failed: {e}")
        summary_rows.append((assay, len(y), int(np.sum(y)), "train_failed"))
        continue

    # evaluate
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = float(accuracy_score(y_test, y_pred))
        bacc = float(balanced_accuracy_score(y_test, y_pred))
        auc  = float(roc_auc_score(y_test, y_prob))
        mcc  = float(matthews_corrcoef(y_test, y_pred))
    except Exception as e:
        print(" Evaluation failed:", e)
        acc = bacc = auc = mcc = 0.0

    # save model
    out_model_path = os.path.join(OUT_DIR, f"{assay}.pkl")
    with open(out_model_path, "wb") as fh:
        pickle.dump(model, fh)

    print(f" Saved model -> {out_model_path}")
    print(f" Results: ACC={acc:.4f}  BACC={bacc:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}")

    summary_rows.append((assay, len(y), int(np.sum(y)), "trained", acc, bacc, auc, mcc))

# -----------------------------
# SAVE SUMMARY
# -----------------------------
cols = ["Assay", "Samples", "Positives", "Status", "ACC", "BACC", "AUC", "MCC"]
# build DataFrame with consistent columns
rows = []
for r in summary_rows:
    if r[3] == "trained":
        rows.append([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]])
    else:
        # pad with empty metrics
        rows.append([r[0], r[1], r[2], r[3], None, None, None, None])

summary_df = pd.DataFrame(rows, columns=cols)
summary_csv = os.path.join(OUT_DIR, "training_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print("\nTraining finished. Summary saved to:", summary_csv)
