import joblib

from fastapi import FastAPI
import pandas as pd

from data_models.input_model import InputData

app = FastAPI(title="ML-OPS APP", version="0.0.1")


model = joblib.load('models/model.pkl')


@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}


@app.post("/predict")
async def predict(input_data: InputData):
    df = pd.DataFrame([
        input_data.model_dump().values()
    ], columns=input_data.model_dump().keys())
    prediction = model.predict(df)
    return {"predicted_class": int(prediction[0])}