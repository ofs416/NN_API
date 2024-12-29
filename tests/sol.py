# File name: model_client.py
import requests

test_molecules = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC1=CC=C(C=C1)O",  # p-Cresol
    "C1=CC=C(C=C1)CCN",  # Phenethylamine
]

for mol in test_molecules:
    response = requests.post("http://127.0.0.1:8000/", params={"smiles": mol})
    print(f"{mol}:", response.json())
