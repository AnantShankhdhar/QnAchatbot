import os
import joblib
import pandas as pd
from fastapi import FastAPI
from qna_model import my_model

index_file = 'data/my_index.faiss'
dataset_dir = 'data/embedded_truefoundry_vectors'

@app.post("/predict")
def predict(
    inp_txt: str
):
    model = my_model(dataset_dir,index_file)
    prediction = my_model(inp_txt)
    return {"prediction": prediction}