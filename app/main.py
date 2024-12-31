import ray
from ray import serve
from fastapi import FastAPI

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from typing import List, Union, Optional
from pydantic import BaseModel


class SolubilityRequest(BaseModel):
    smiles: str


class BatchSolubilityRequest(BaseModel):
    smiles_list: List[str]


class SolubilityResponse(BaseModel):
    smiles: str
    solubility: Optional[float] = None
    error: Optional[str] = None


class BatchSolubilityResponse(BaseModel):
    predictions: List[SolubilityResponse]


app = FastAPI(title="SMILES Solubility Prediction Service")


@serve.deployment(
    num_replicas="auto", ray_actor_options={"num_cpus": 0.2, "num_gpus": 0}
)
@serve.ingress(app)
class SolubilityInference:
    def __init__(
        self, model_path: str = "models/model_scripted.pt", device: str = None
    ):
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

    def predict_single(self, smiles: str):
        """Predict solubility for a single SMILES string."""
        features = self.process_smiles(smiles)
        if features is None:
            return SolubilityResponse(
                smiles=smiles, solubility=None, error="Could not process SMILES string"
            )

        try:
            with torch.no_grad():
                features_tensor = (
                    torch.FloatTensor(features).unsqueeze(0).to(self.device)
                )
                prediction = self.model(features_tensor).squeeze().item()
                return SolubilityResponse(smiles=smiles, solubility=prediction)
        except Exception as e:
            return SolubilityResponse(
                smiles=smiles, solubility=None, error=f"Prediction error: {str(e)}"
            )

    @app.post("/predict")
    async def predict_single_endpoint(self, request: SolubilityRequest):
        return self.predict_single(request.smiles)

    @app.post("/predict_batch")
    async def predict_batch(self, request: BatchSolubilityRequest):
        """Endpoint for batch SMILES prediction"""
        predictions = []
        for smiles in request.smiles_list:
            pred = self.predict_single(smiles)
            predictions.append(pred)
        return BatchSolubilityResponse(predictions=predictions)


SolubilityInference_app = SolubilityInference.bind()


if __name__ == "__main__":
    # Connect to the running Ray Serve instance.
    ray.init(address="auto", namespace="serve-example", ignore_reinit_error=True)
    serve.run(SolubilityInference_app, route_prefix="/hello")
