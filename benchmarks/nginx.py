import subprocess
import time
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor

"""Comparison of scale timings. Naive implementation with limited executor speed hence the 
differences aren't as large as expected."""


def run_docker_compose(scale):
    # Stop any running containers
    subprocess.run(["docker", "compose", "down"], check=True)

    # Start with specified scale
    subprocess.run(
        ["docker", "compose", "up", "-d", "--scale", f"api={scale}"], check=True
    )

    # Wait for services to be ready
    time.sleep(20)  # Give some time for services to start


def make_request():
    # Create example matrices
    matrix1 = np.array([[1, 2], [3, 4]])
    matrix2 = np.array([[5, 6], [7, 8]])

    data = {"matrix1": matrix1.tolist(), "matrix2": matrix2.tolist()}

    response = requests.post("http://127.0.0.1:8080//matrix/multiply", json=data)
    return


def run_benchmark(num_requests=1_000):
    # Use ThreadPoolExecutor to send concurrent requests
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
    return


def main():
    # Test with scale=3
    print("\nStarting test with scale=3...")
    run_docker_compose(scale=3)
    start_time = time.time()
    run_benchmark()
    time_taken_3 = time.time() - start_time
    print(f"\nResults for scale=3:")
    print(f"Total time: {time_taken_3:.2f} seconds")

    # Test with scale=1
    print("\nStarting test with scale=1...")
    run_docker_compose(scale=1)
    start_time = time.time()
    run_benchmark()
    time_taken_1 = time.time() - start_time
    print(f"\nResults for scale=1:")
    print(f"Total time: {time_taken_1:.2f} seconds")

    # Calculate percentage difference
    time_diff_percent = (time_taken_1 - time_taken_3) / time_taken_3 * 100

    print(f"\nComparison:")
    print(f"Scale=3 vs Scale=1 difference: {time_diff_percent:.1f}%")

    # Clean up
    subprocess.run(["docker", "compose", "down"], check=True)


if __name__ == "__main__":
    main()
