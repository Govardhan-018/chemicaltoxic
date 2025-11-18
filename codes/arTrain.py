import pandas as pd
import glob
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import swifter

# Reduce RDKit console logging. `swifter` speeds up pandas.apply for large dataframes.

# Load SMILES file (two columns expected: SMILES and ID)
smiles_df = pd.read_csv(
    "/home/hegde_go/Documents/AIML-LAB-EL/tox21_10k_challenge_score.smiles",
    sep=r"\s+",
    names=["SMILES", "ID"],
    engine="python",
    comment="#"
)

# Load label TSV files from the AR directory and concatenate them
label_files = glob.glob("/home/hegde_go/Documents/AIML-LAB-EL/tox21-challenge/AR/*.txt")

label_list = []
for file in label_files:
    df = pd.read_csv(file, sep="\t")
    label_list.append(df)

labels_df = pd.concat(label_list, ignore_index=True)

# Merge SMILES and label tables on the sample ID and keep relevant columns
merged = smiles_df.merge(labels_df, left_on="ID", right_on="Sample ID")
merged = merged[["ID", "SMILES", "Activity"]]

print("Before dropping NaN:", len(merged))

# Drop samples without activity labels and ensure integer labels
merged = merged.dropna(subset=["Activity"])
merged["Activity"] = merged["Activity"].astype(int)

print("After dropping NaN:", len(merged))


# Featurization: convert SMILES to 2048-bit Morgan fingerprints (radius=2)
def featurize(smi):
    """Return a 2048-bit Morgan fingerprint (radius=2) as a Python list.

    Returns None when the SMILES string cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return list(fp)

# Compute fingerprints in parallel with swifter and drop any failed conversions
merged["FP"] = merged["SMILES"].swifter.apply(featurize)
merged = merged.dropna(subset=["FP"])  # remove failed featurizations

# Prepare training arrays (X: list of fingerprint vectors, y: integer labels)
X = list(merged["FP"])
y = merged["Activity"]


# Train a Random Forest classifier on the fingerprint features
model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1  # use all available CPU cores
)
model.fit(X, y)

print("Training done!")

# Save the trained model to disk
with open("tox_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as tox_model.pkl")
