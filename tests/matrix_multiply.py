import numpy as np
import requests

# Create example matrices using NumPy
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
expected_matrix = np.array([[19., 22.], [43., 50.]])


# Convert NumPy arrays to lists for JSON serialization
data = {"matrix1": matrix1.tolist(), "matrix2": matrix2.tolist()}

# Send the POST request
url = "http://127.0.0.1:8080//matrix/multiply"
response = requests.post(url, json=data)

result = response.json()
matrix_result = np.array(result["result_array"])

np.testing.assert_array_equal(expected_matrix, matrix_result)
