# Hand Sign Recognition System (A, B, C, D)

This project is a Hand Sign Recognition Web Application that detects hand landmarks using MediaPipe, classifies hand signs using a TensorFlow/Keras model, and serves predictions through a FastAPI backend.
The system currently supports recognition of four hand signs: A, B, C, and D.

# Features

-Real-time hand detection using webcam
-Hand landmark extraction using MediaPipe Hands (21 landmarks)
-Deep Learning model for hand sign classification
-FastAPI backend for prediction API
-Simple HTML + JavaScript frontend
-Manual input and live video prediction supported

# Supported Hand Signs

A
B
C
D
Note: The model is trained only on these four classes. Predictions outside these signs are not supported.

# Model Details

Input features: 42
21 x-coordinates
21 y-coordinates
Input format:
[x0, x1, ..., x20, y0, y1, ..., y20]
Normalization:
x coordinates normalized by frame width
y coordinates normalized by frame height
Framework: TensorFlow / Keras
Model file: handsignmodel.keras

# Install Dependencies

pip install fastapi uvicorn tensorflow numpy pydantic
Run the Server
uvicorn fast:app --reload
Server will start at:
http://127.0.0.1:8000

# Frontend (index.html)
Features
-Webcam-based live prediction
-Manual input of 42 landmark values
-Canvas visualization of hand landmarks
How to Run
-Simply open index.html in a browser
-Allow camera access
-Ensure FastAPI server is running

# Dataset Preparation

-Individual CSV files were created for each gesture
-A label column was added to each dataset
-All datasets were merged into a single merged.csv file using merging_data.py

# Testing

-Use testing.py to:
-Normalize raw landmark coordinates
-Verify model predictions without frontend

# Author
Ch. Mounika
B.Tech Student
Hand Sign Recognition Project using Deep Learning(CNN)
