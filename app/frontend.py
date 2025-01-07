import streamlit as st
import requests

st.text_input("Molecule SMILE", value = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

# You can access the value at any point with:
if st.session_state.name is str:
    response = requests.post(
            "http://127.0.0.1:8000/predict", json={"smiles": st.session_state.name}
        ).json()
elif st.session_state.name is list[str]:
    response = requests.post(
            "http://127.0.0.1:8000/predict_batch", json={"smiles_list": st.session_state.name}
        ).json()
else:
    response = None

st.write(response)