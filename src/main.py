from fastapi import FastAPI
from src.predict import analyze_complaint
from schemas.complaint import ComplaintRequest, ComplaintResponse

app= FastAPI()

@app.get('/')
def home():
    return {"message": "AI Complaint System Running 🚀"}

@app.post('analyze', response_model= ComplaintResponse)
def analyze(req: ComplaintRequest):
    result= analyze_complaint(req.text)
    return result