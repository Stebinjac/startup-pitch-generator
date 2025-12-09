from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = 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
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://startup-pitch-generator.vercel.app"],  # You can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PitchRequest(BaseModel):
    idea: str
    tone: str
    audience: str
    industry: str

@app.post("/generate_pitch")
def generate_pitch(data: PitchRequest):
    prompt = (
        f"Generate a detailed 200+ word startup pitch for the following:\n\n"
        f"Idea: {data.idea}\n"
        f"Industry: {data.industry}\n"
        f"Tone: {data.tone}\n"
        f"Target Audience: {data.audience}\n\n"
        f"The pitch should clearly outline the problem, solution, product features, and potential impact."
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        res_data = response.json()

        # Debug log if something goes wrong
        if "choices" not in res_data:
            print("Groq API Error:", res_data)
            return {"error": "Failed to generate pitch", "details": res_data}

        pitch_text = res_data["choices"][0]["message"]["content"]
        return {"pitch": pitch_text}

    except Exception as e:
        print("Exception:", e)
        return {"error": "An error occurred while generating pitch."}

