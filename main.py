from fastapi import FastAPI
import pickle
import numpy as np
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    cgpa: float
    iq: int
    profile_score: int

class OutputData(BaseModel):
    result: str

# Load your pre-trained model from a pickle file
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the prediction API!"}

@app.post("/predict")
async def predict_data(data: InputData):

    # prediction
    pred = model.predict(np.array([data.cgpa, data.iq, data.profile_score]).reshape(1,3))
    print('result: ', pred)
    if pred[0] == 1:
        result = 'placed'
    else:
        result = 'not placed'

    return {'result': result}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)

#uvicorn main:app --reload


