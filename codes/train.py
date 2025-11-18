import os
import pandas as pd
import glob
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import swifter

# -----------------------------
# Path Settings
# -----------------------------
SMILES_PATH = "/home/hegde_go/Documents/AIML-LAB-EL/tox21_10k_challenge_score.smiles"
LABEL_ROOT = "/home/hegde_go/Documents/AIML-LAB-EL/tox21-challenge"

# -----------------------------
# Load SMILES file once
# -----------------------------
smiles_df = pd.read_csv(
    SMILES_PATH,
    sep=r"\s+",
    names=["SMILES", "ID"],
    engine="python",
    comment="#"
)

print("Loaded SMILES:", len(smiles_df))

# -----------------------------
# Morgan fingerprint function
# -----------------------------
def featurize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return list(fp)

# -----------------------------
# Function to process each assay
# -----------------------------
def process_assay(folder):
    print(f"\n🔹 Processing assay: {folder}")

    folder_path = os.path.join(LABEL_ROOT, folder)
    label_files = glob.glob(folder_path + "/*.txt")

    if len(label_files) == 0:
        print(f"⚠️ No label files found in {folder}, skipping.")
        return

    # Load labels
    label_list = []
    for file in label_files:
        df = pd.read_csv(file, sep="\t")
        label_list.append(df)

    labels_df = pd.concat(label_list, ignore_index=True)

    # Detect ID column
    possible_ids = ["Sample ID", "sample_id", "ID", "Molecule ID", "Compound ID"]
    label_id = next((c for c in possible_ids if c in labels_df.columns), None)

    if label_id is None:
        print(f"⚠️ No ID column found for {folder}, skipping.")
        return

    # Merge SMILES with labels
    merged = smiles_df.merge(labels_df, left_on="ID", right_on=label_id, how="inner")

    # Safety check
    if "ID" not in merged.columns or "SMILES" not in merged.columns:
        print(f"⚠️ Invalid columns after merge for {folder}, skipping.")
        return

    # Detect Activity column
    if "Activity" in merged.columns:
        activity_col = "Activity"
    elif "activity" in merged.columns:
        activity_col = "activity"
    else:
        print(f"⚠️ No Activity column in {folder}, skipping.")
        return

    merged = merged[["ID", "SMILES", activity_col]]
    merged.rename(columns={activity_col: "Activity"}, inplace=True)

    # Drop NaN / convert type
    merged = merged.dropna(subset=["Activity"])
    merged["Activity"] = merged["Activity"].astype(int)

    if len(merged) == 0:
        print(f"⚠️ No usable samples for {folder}, skipping.")
        return

    # Featurization
    print(f"🔧 Featurizing molecules for {folder}...")
    merged["FP"] = merged["SMILES"].swifter.apply(featurize)
    merged = merged.dropna(subset=["FP"])

    X = list(merged["FP"])
    y = merged["Activity"]

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Train model
    print(f"🚀 Training model for {folder} ...")
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # -----------------------------
    # 🟢 Evaluation
    # -----------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n Performance for {folder}:")
    print(f"   ➤ Samples: {len(merged)}")
    print(f"   ➤ Activity distribution: {y.value_counts().to_dict()}")
    print(f"   ➤ Accuracy: {acc:.4f}")
    print(f"   ➤ ROC-AUC: {auc:.4f}")

    # Save model
    out_name = f"{folder}_model.pkl"
    with open(out_name, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Saved model: {out_name}")

# -----------------------------
# Train all assays sequentially
# -----------------------------
print("\n===============================")
print(" Training all assays sequentially ")
print("===============================\n")

folders = sorted([
    d for d in os.listdir(LABEL_ROOT)
    if os.path.isdir(os.path.join(LABEL_ROOT, d))
])

for folder in folders:
    process_assay(folder)

print("\n🎉 All models trained successfully (SEQUENTIAL MODE)!")
