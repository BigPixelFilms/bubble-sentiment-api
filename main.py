from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Bubble Sentiment API")

# Load model once
model = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Sentiment API is running"}


@app.post("/analyze")
def analyze_sentiment(data: TextInput):

    result = model(data.text)[0]

    return {
        "sentiment": result["label"],
        "confidence": round(float(result["score"]) * 100, 2)
    }
