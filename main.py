from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

#Constantes(entrenamiento)
MAX_PESO = 119.9
MIN_PESO = 45.0
MAX_ESTATURA = 2.0
MIN_ESTATURA = 1.5
#Procesamiento de datos
def min_max(input, min, max):
  return (input - min)/(max-min)


def process_data(data):
    data.genero = 1 if data.genero == "M" else 0
    data.estatura = min_max(data.estatura, MIN_ESTATURA, MAX_ESTATURA)
    data.peso = min_max(data.peso, MIN_PESO, MAX_PESO)
    return data
app = FastAPI(title="Imc with ML")

model = joblib.load("imc_gbr.pkl")

class InputData(BaseModel):
    genero: str
    estatura: float
    peso: float

@app.get("/")
def root():
    return {"status": "running"}
@app.post("/predict")
def predict(data: InputData):
    data = process_data(data)
    x = np.array([data.peso, data.estatura, data.genero]).reshape(1, -1)
    pred = model.predict(x)
    return {"predicción: ": float(pred[0])}