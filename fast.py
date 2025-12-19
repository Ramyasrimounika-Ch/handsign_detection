from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tensorflow as tf
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or use ["http://127.0.0.1:5500"] for stricter rule
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load the trained model
model = tf.keras.models.load_model("handsignmodel.keras")

NUM_FEATURES = 42  # 21 x-coordinates + 21 y-coordinates
CLASS_NAMES = ['A', 'B', 'C', 'D']  # update as needed

class InputData(BaseModel):
    data: List[float]

@app.get("/")
def root():
    return {"message": "Handsign prediction API is running."}

@app.post("/predict")
def predict(input: InputData):
    if len(input.data) != NUM_FEATURES:
        return {"error": f"Expected {NUM_FEATURES} features, but got {len(input.data)}."}
    
    input_array = np.array([input.data])
    prediction = model.predict(input_array)
    
    predicted_index = int(np.argmax(prediction))
    predicted_label = CLASS_NAMES[predicted_index]

    return {
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
    }
