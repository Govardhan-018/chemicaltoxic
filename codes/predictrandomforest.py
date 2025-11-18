import pickle
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')

MODEL_DIR = "./modelrandomforest"   # folder where RF models are stored

# --------------------------------------------------
#  FEATURE GENERATION — must match RandomForest training
# --------------------------------------------------
def featurize(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None

        # ONLY Morgan 2048 bits (same as training)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(fp, dtype=np.float32)
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

    prob = model.predict_proba(feat)[0][1]
    pred = model.predict(feat)[0]

    return int(pred), float(prob)

# --------------------------------------------------
#  LOAD ALL MODELS
# --------------------------------------------------
def load_models():
    models = {}
    print("Loading models from:", MODEL_DIR)

    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            name = f.replace(".pkl", "")
            path = os.path.join(MODEL_DIR, f)

            # FIX: open the file before pickle.load()
            with open(path, "rb") as file:
                models[name] = pickle.load(file)

            print(f"✔ Loaded {name}")

    if not models:
        print("❌ No models found!")
        exit()

    print("\nAll models loaded!\n")
    return models

# --------------------------------------------------
#  INTERACTIVE LOOP
# --------------------------------------------------
if __name__ == "__main__":
    models = load_models()

    print("=====================================")
    print("   🔬 TOX21 MULTI-ASSAY PREDICTOR")
    print("=====================================\n")

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
                print(f"👉 {assay}: (error)\n")
                continue

            tox = "Toxic" if pred == 1 else "Non-Toxic"

            print(f"👉 {assay}")
            print(f"   Toxicity: {tox}")
            print(f"   Probability: {prob:.4f}\n")

        print("--------------------------------------\n")
