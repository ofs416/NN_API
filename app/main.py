from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
from typing import List


class MatricesInput(BaseModel):
    matrix1: List[List[float]]
    matrix2: List[List[float]]


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "API is running"}


@app.post("/matrix/multiply")
async def multiply_matrices(matrices: MatricesInput):
    # Convert inputs to numpy arrays
    matrix1 = np.array(matrices.matrix1)
    matrix2 = np.array(matrices.matrix2)

    # Check if multiplication is possible
    if matrix1.shape[1] != matrix2.shape[0]:
        return {"error": "Matrix dimensions don't match for multiplication"}

    # Perform multiplication
    result = np.matmul(matrix1, matrix2)

    # Format the result as a string in numpy style
    np.set_printoptions(precision=4, suppress=True)  # Suppress scientific notation

    return {
        "result_array": result.tolist(),  # Original format for further processing
    }
