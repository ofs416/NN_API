from locust import HttpUser, task

class HelloWorldUser(HttpUser):
    @task
    def hello_world(self):
        self.client.post("http://127.0.0.1:8000/predict", json={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"})
