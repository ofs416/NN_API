import requests
import numpy as np

# Create two numpy arrays
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Convert to lists for JSON serialization
data = {
    "matrix1": matrix1.tolist(),
    "matrix2": matrix2.tolist()
}

# Make the request
response = requests.post("http://localhost:8000/matrix/multiply", json=data)
result = response.json()

# If you want to convert back to numpy array and print:
array_result = np.array(result["result_array"])
print("NumPy array:")
print(array_result)