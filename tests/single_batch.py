import requests

test_molecules = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC1=CC=C(C=C1)O",  # p-Cresol
    "C1=CC=C(C=C1)CCN",  # Phenethylamine
]

for mol in test_molecules:
    response = requests.post("http://127.0.0.1:8000/predict", json={"smiles": mol})
    print(response.json())


data = {"smiles_list": test_molecules}
response = requests.post(
    "http://127.0.0.1:8000/predict_batch", json=data
)
for i in response.json()["predictions"]:
    print(i)
