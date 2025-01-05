import requests

test_molecules = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC1=CC=C(C=C1)O",  # p-Cresol
    "C1=CC=C(C=C1)CCN",  # Phenethylamine
]

data = {"smiles_list": test_molecules}
response_batch = requests.post("http://127.0.0.1:8000/predict_batch", json=data)
for i, response in enumerate(response_batch.json()["predictions"]):
    print(response)
    response_single = requests.post(
            "http://127.0.0.1:8000/predict", json={"smiles": test_molecules[i]}
            ).json()
    print(response_single)
    assert response == response_single
