import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import os

# Disable RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# -----------------------------
# Morgan fingerprint
# -----------------------------
def featurize(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return list(fp)

# -----------------------------
# Predict toxicity using a model
# -----------------------------
def predict(model, smile):
    fp = featurize(smile)
    if fp is None:
        return None, None
    prob = model.predict_proba([fp])[0][1]
    pred = model.predict([fp])[0]
    return pred, prob

# -----------------------------
# Load all models automatically
# -----------------------------
models = {}

print("=====================================")
print("   🔬 TOX21 MULTI-ASSAY PREDICTOR")
print("=====================================\n")

print("Loading models...\n")

for f in os.listdir():
    if f.endswith("_model.pkl"):
        assay = f.replace("_model.pkl", "")
        with open(f, "rb") as file:
            models[assay] = pickle.load(file)
        print(f"✔ Loaded {assay}")

print("\nAll models loaded successfully!")
print("Type 'exit' to stop.\n")

# -----------------------------
# Continuous input loop
# -----------------------------
while True:
    smile = input("\nEnter SMILES: ")

    if smile.lower() == "exit":
        print("👋 Exiting. Goodbye!")
        break

    fp = featurize(smile)
    if fp is None:
        print("❌ Invalid SMILES! Try again.")
        continue

    print("\n=======================")
    print("  🧪 Toxicity Results")
    print("=======================\n")

    for assay, model in models.items():
        pred, prob = predict(model, smile)
        tox = "Toxic" if pred == 1 else "Non-Toxic"

        print(f"👉 {assay}")
        print(f"   Toxicity: {tox}")
        print(f"   Probability: {prob:.4f}\n")

    print("--------------------------------------")
