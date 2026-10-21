# Use the official Python image as the base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Set work directory
WORKDIR /code

# Install dependencies
COPY requirements.txt /code/

RUN python -m pip install --upgrade pip

RUN python -m pip install \
    "mlflow==2.22.5" \
    "numpy==2.2.6" \
    "pandas==2.2.3" \
    "scikit-learn==1.6.1" \
    joblib \
    azure-storage-blob \
    azure-identity \
    azure-ai-ml \
    azureml-mlflow \
    fastapi \
    uvicorn \
    python-dotenv \
    pytest

RUN python -m pip check
# Copy project
COPY ./app ./

# Expose the port FastAPI will run on
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]