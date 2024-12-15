# Matrix Multiplication API

A FastAPI-based web service that performs matrix multiplication using NumPy. This service is containerized using Docker and served through Nginx.

## Features

- Matrix multiplication endpoint
- JSON request/response format
- Docker containerization
- Nginx reverse proxy
- Development mode with hot reloading
- Production-ready configuration

## Prerequisites

- Docker
- Docker Compose

## Project Structure

```
.
├── app/
│   └── main.py              # FastAPI application
├── dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── nginx.conf              # Nginx configuration
├── pyproject.toml          # Python project dependencies
├── uv.lock                 # Dependency lock file
└── README.md
```

## Installation & Running

### Development Mode

Run the application with hot reloading enabled:

```bash
docker compose up --build
```

### Production Mode

Run the application in production mode:

```bash
docker compose -f docker-compose.prod.yml up --build
```

## API Documentation

### Matrix Multiplication Endpoint

**Endpoint**: `/matrix/multiply`  
**Method**: POST  
**Content Type**: application/json

**Request Body**:
```json
{
    "matrix1": [[1, 2], [3, 4]],
    "matrix2": [[5, 6], [7, 8]]
}
```

**Response**:
```json
{
    "result_array": [[19, 22], [43, 50]]
}
```

### Health Check

**Endpoint**: `/`  
**Method**: GET

**Response**:
```json
{
    "message": "API is running test"
}
```

## Testing

You can test the API using the provided test script:

```python
import numpy as np
import requests

# Create example matrices
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Convert NumPy arrays to lists for JSON serialization
data = {
    "matrix1": matrix1.tolist(),
    "matrix2": matrix2.tolist()
}

# Send the POST request
url = "http://localhost:8080/matrix/multiply"
response = requests.post(url, json=data)

# Print the result
result = response.json()
print("Result:", result["result_array"])
```

## Development

The project uses:
- FastAPI for the web framework
- NumPy for matrix operations
- uv for dependency management
- Docker for containerization
- Nginx as a reverse proxy

### Development Dependencies

Development dependencies are managed in `pyproject.toml`:
- matplotlib
- pandas
- ruff (for linting)

## Configuration


### Ports

- 8080

### Docker Resources

The production configuration includes resource limits:
- API: 2 CPU cores, 1GB memory
- Nginx: 1 CPU core, 512MB memory


## License

This project is open source and available under the MIT License.