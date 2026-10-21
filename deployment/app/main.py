from fastapi import FastAPI, Request
import mlflow.pyfunc
import pandas as pd
import os
from contextlib import asynccontextmanager
import time
from fastapi.responses import PlainTextResponse
import os
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
    model_uri = os.getenv("MODEL_URI")
    if not model_uri:
        raise ValueError("MODEL_URI environment variable is not set.")
    model = mlflow.pyfunc.load_model(f"models:/{model_uri}/latest")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    
    return {"prediction": prediction.tolist()}