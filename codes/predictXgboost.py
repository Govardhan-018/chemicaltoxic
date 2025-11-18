import pickle
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')

MODEL_DIR = "./modelsxgboost"   # <── CHANGED HERE


# --------------------------------------------------
#  FEATURE GENERATION — must match training features
# --------------------------------------------------
def featurize(smile):
    """
    Generate a 2054-length feature vector:
    - 2048-bit Morgan fingerprint
    - 6 physicochemical descriptors
    """
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
    except:
        return None


# --------------------------------------------------
#  PREDICT (single model)
# --------------------------------------------------
def predict(model, smile):
    feat = featurize(smile)
    if feat is None:
        return None, None

    feat = feat.reshape(1, -1)

    prob = float(model.predict_proba(feat)[0][1])
    pred = int(model.predict(feat)[0])

    return pred, prob


# --------------------------------------------------
#  LOAD ALL MODELS FROM modelxgboost/ FOLDER
# --------------------------------------------------
def load_models():
    models = {}
    print("Loading models from folder:", MODEL_DIR)

    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            name = f.replace(".pkl", "")
            path = os.path.join(MODEL_DIR, f)

            with open(path, "rb") as fh:
                models[name] = pickle.load(fh)

            print(f"✔ Loaded {name}")

    if not models:
        print("❌ No model files found in", MODEL_DIR)
        exit()

    print("\nAll models loaded successfully!\n")
    return models


# --------------------------------------------------
#  INTERACTIVE LOOP
# --------------------------------------------------
if __name__ == "__main__":
    models = load_models()

    print("=====================================")
    print("   🔬 TOX21 MULTI-ASSAY PREDICTOR")
    print("=====================================\n")
    print("Enter SMILES to predict toxicity across all assays.")
    print("Type 'exit' to quit.\n")

    while True:
        smi = input("Enter SMILES: ").strip()

        if smi.lower() == "exit":
            print("👋 Goodbye!")
            break

        feat = featurize(smi)
        if feat is None:
            print("❌ Invalid SMILES! Try again.\n")
            continue

        print("\n=======================")
        print("  🧪 Toxicity Results")
        print("=======================\n")

        for assay, model in models.items():
            pred, prob = predict(model, smi)

            if pred is None:
                print(f"👉 {assay}: ❌ Prediction failed")
                continue

            tox_label = "Toxic" if pred == 1 else "Non-Toxic"

            print(f"👉 {assay}")
            print(f"   Toxicity: {tox_label}")
            print(f"   Probability: {prob:.4f}\n")

        print("--------------------------------------\n")
