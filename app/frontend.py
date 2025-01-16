import streamlit as st
import requests
from typing import Union, List, Optional
import json

def predict_molecule(smiles: Union[str, List[str]]) -> Optional[dict]:
    """
    Send SMILES string(s) to prediction API and return response.
    
    Args:
        smiles: Single SMILES string or list of SMILES strings
        
    Returns:
        API response as dictionary or None if request fails
    """
    try:
        if isinstance(smiles, str):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"smiles": smiles}
            )
        elif isinstance(smiles, list):
            response = requests.post(
                "http://127.0.0.1:8000/predict_batch",
                json={"smiles_list": smiles}
            )
        else:
            st.error(f"Invalid input type: {type(smiles)}. Expected str or list[str].")
            return None
            
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None
    except json.JSONDecodeError:
        st.error("Failed to decode API response")
        return None

def main():
    st.title("Molecule Predictor")
    
    # Initialize session state
    if "molecule_input" not in st.session_state:
        st.session_state.molecule_input = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    
    # Get SMILES input
    smiles_input = st.text_input(
        "Enter Molecule SMILES",
        key="molecule_input",  # This automatically handles session state
        help="Enter a single SMILES string or multiple SMILES separated by commas"
    )
    
    if smiles_input:
        # Process input - split on commas if multiple SMILES provided
        smiles_list = [s.strip() for s in smiles_input.split(",")]
        
        # Determine if single or batch prediction
        if len(smiles_list) == 1:
            prediction = predict_molecule(smiles_list[0])
        else:
            prediction = predict_molecule(smiles_list)
            
        # Display results
        if prediction:
            st.subheader("Prediction Results")
            st.json(prediction)

if __name__ == "__main__":
    main()
