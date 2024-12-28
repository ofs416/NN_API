import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from typing import List, Union


class SolubilityInference:
    def __init__(
        self, model_path: str = "model_scripted.pt", device: str = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load the TorchScript model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def process_smiles(self, smiles: str) -> Union[np.ndarray, None]:
        """Convert SMILES to Morgan fingerprint."""
        # Handle ionic compounds
        if "." in smiles:
            parts = smiles.split(".")
            smiles = max(parts, key=len)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        morgan_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2)
        return morgan_gen.GetFingerprintAsNumPy(mol)

    def predict_single(self, smiles: str) -> Union[float, None]:
        """Predict solubility for a single SMILES string."""
        features = self.process_smiles(smiles)
        if features is None:
            print(f"Could not process SMILES: {smiles}")
            return None

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            prediction = self.model(features_tensor).squeeze().item()
        return prediction

    def predict_batch(self, smiles_list: List[str]) -> List[Union[float, None]]:
        """Predict solubility for a list of SMILES strings."""
        predictions = []
        for smiles in smiles_list:
            pred = self.predict_single(smiles)
            predictions.append(pred)
        return predictions


def main():
    # Initialize the inference class
    predictor = SolubilityInference()

    # Example SMILES strings
    test_molecules = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC1=CC=C(C=C1)O",  # p-Cresol
        "C1=CC=C(C=C1)CCN",  # Phenethylamine
    ]
    predictions = predictor.predict_batch(test_molecules)

    # Print results
    print("\nPrediction Results:")
    print("-" * 50)
    print(f"{'SMILES':<30} | {'Predicted Solubility':<20}")
    print("-" * 50)
    for smiles, pred in zip(test_molecules, predictions):
        if pred is not None:
            print(f"{smiles:<30} | {pred:>20.3f}")
        else:
            print(f"{smiles:<30} | {'Failed to process':>20}")


if __name__ == "__main__":
    main()
