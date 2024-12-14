import numpy as np
import requests

# Create example matrices using NumPy
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Convert NumPy arrays to lists for JSON serialization
data = {"matrix1": matrix1.tolist(), "matrix2": matrix2.tolist()}

# Send the POST request
url = "http://localhost:8000/matrix/multiply"
response = requests.post(url, json=data)

result = response.json()
array_result = np.array(result["result_array"])
print("NumPy array:")
print(array_result)
