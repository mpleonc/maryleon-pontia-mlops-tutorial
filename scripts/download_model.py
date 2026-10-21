import mlflow
import os

MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000/")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "hola5-adult-income")
mlflow.set_tracking_uri(MLFLOW_URI)
# mlflow.set_tracking_uri("azureml://spaincentral.api.azureml.ms/mlflow/v1.0/subscriptions/84261411-9013-44d1-b8e1-6ba904def1d1/resourceGroups/rg-alberto-pontia/providers/Microsoft.MachineLearningServices/workspaces/aml-curso-devops-2026")

local_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{EXPERIMENT_NAME}/latest",
    dst_path="./models"
)

print(f"Modelo descargado en: {local_path}")