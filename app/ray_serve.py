import ray
from ray import serve
from fastapi import FastAPI

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from typing import List, Union


app = FastAPI()


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.5, "num_gpus": 0})
@serve.ingress(app)
class SolubilityInference:
    def __init__(self, model_path: str = "models/model_scripted.pt", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load the TorchScript model
        self.model = torch.jit.load(model_path, map_location=self.device)

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

    @app.post("/")
    def predict_single(self, smiles: str):
        """Predict solubility for a single SMILES string."""
        features = self.process_smiles(smiles)
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            prediction = self.model(features_tensor).squeeze().item()
        return prediction


SolubilityInference_app = SolubilityInference.bind()





