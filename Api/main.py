from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import json
from utils import preprocess,vectorize_log
import csv
from fastapi.responses import FileResponse,JSONResponse
import joblib
from pydantic import BaseModel
#Define your FastAPI app
app = FastAPI()


#Create an endpoint to use the saved model
@app.post("/predict")
def predict(log: str):
    # Load the saved model
    loaded_model = joblib.load('risky_safe_model.pkl')
    # Preprocess the log
    log = preprocess(log)
    # Vectorize the log using the model and reshape the vectorized log to be a 2D array 
    log = vectorize_log(log).reshape(1, -1)
    # Make prediction 
    # Print the shape after vectorization
    print("Log shape after vectorization:", log.shape)
    prediction = loaded_model.predict(log)
    # Return the prediction as a json
      # Determine the response message based on the prediction
    if prediction[0] == 0:
        #parse prediction to a json object
        result = {"prediction": int(prediction[0]), "message": "Safe log"}

    else:
        result = {"prediction": int(prediction[0]), "message": "Risky log"}
    print("Result:", result)
    return JSONResponse(result)
    