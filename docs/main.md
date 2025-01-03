Module main
===========

Classes
-------

`BatchSolubilityRequest(**data: Any)`
:   Request for batch solubility prediction
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel

    ### Class variables

    `model_config`
    :

    `smiles_list: List[str]`
    :   List of SMILES strings

`BatchSolubilityResponse(**data: Any)`
:   Response for batch solubility prediction
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel

    ### Class variables

    `model_config`
    :

    `predictions: List[main.SolubilityResponse]`
    :   List of solubility predictions

`SolubilityRequest(**data: Any)`
:   Request for solubility prediction
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel

    ### Class variables

    `model_config`
    :

    `smiles: str`
    :   SMILES string of the molecule

`SolubilityResponse(**data: Any)`
:   Response for solubility prediction
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel

    ### Class variables

    `error: str | None`
    :   Error message

    `model_config`
    :

    `smiles: str`
    :   SMILES string of the molecule

    `solubility: float | None`
    :   Predicted solubility value