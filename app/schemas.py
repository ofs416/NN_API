from pydantic import BaseModel
from typing import List


class MatricesInput(BaseModel):
    matrix1: List[List[float]]
    matrix2: List[List[float]]