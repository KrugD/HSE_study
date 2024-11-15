import dill
import uvicorn
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import re
import numpy as np

app = FastAPI()

with open('model/price_cars.pkl', 'rb') as file:
    model = dill.load(file)

with open('model/cars_pipeline.pkl', 'rb') as file:
    pipeline = dill.load(file)

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

@app.get('/status')
def status():
    return 'I am OK'

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    input_data = pipeline['pipeline'].transform(pd.DataFrame([item.dict()]))
    prediction = model['model'].predict(input_data)
    return prediction[0]


@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    input_data = pipeline['pipeline'].transform(df)
    predictions = model['model'].predict(input_data)
    df['predicted_price'] = predictions

    output_file = "Predictions/predictions.csv"
    df.to_csv(output_file, index=False)

    return {"message": "Предсказания успешно сохранены", "output_file": output_file}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)