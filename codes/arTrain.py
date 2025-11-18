import pandas as pd
import glob
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import swifter

# -----------------------------
# 1. Load SMILES file 
# -----------------------------
smiles_df = pd.read_csv(
    "/home/hegde_go/Documents/AIML-LAB-EL/tox21_10k_challenge_score.smiles",
    sep=r"\s+",
    names=["SMILES", "ID"],
    engine="python",
    comment="#"
)

# -----------------------------
# 2. Load ALL label files in AR folder
# -----------------------------
label_files = glob.glob("/home/hegde_go/Documents/AIML-LAB-EL/tox21-challenge/AR/*.txt")

label_list = []
for file in label_files:
    df = pd.read_csv(file, sep="\t")
    label_list.append(df)

labels_df = pd.concat(label_list, ignore_index=True)

# -----------------------------
# 3. Merge SMILES + Labels
# -----------------------------
merged = smiles_df.merge(labels_df, left_on="ID", right_on="Sample ID")
merged = merged[["ID", "SMILES", "Activity"]]

print("Before dropping NaN:", len(merged))

# ❗ Remove unlabeled samples
merged = merged.dropna(subset=["Activity"])

# Convert to int
merged["Activity"] = merged["Activity"].astype(int)

print("After dropping NaN:", len(merged))

# -----------------------------
# 4. Convert SMILES to Morgan Fingerprints
# -----------------------------
def featurize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return list(fp)

# 🔥 FAST PARALLEL FEATURE GENERATION
merged["FP"] = merged["SMILES"].swifter.apply(featurize)

merged = merged.dropna(subset=["FP"])

X = list(merged["FP"])
y = merged["Activity"]

# -----------------------------
# 5. Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1   # 🔥 Train using ALL CPU CORES
)
model.fit(X, y)

print("Training done!")

# -----------------------------
# 6. Save final model
# -----------------------------
with open("tox_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as tox_model.pkl")
