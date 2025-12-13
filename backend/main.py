from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEYY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing. Set it in Render (or local .env).")

MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI()

# --- CORS ---
# For quick debugging you can use ["*"], but replace with your exact frontend origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- temporarily "*" for testing, change to ["https://your-frontend.com"] for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PitchRequest(BaseModel):
    idea: str
    tone: str
    audience: str
    industry: str

# Simple root so / doesn't 404
@app.get("/")
def root():
    return {"status": "ok", "message": "Pitch API running. See /docs for API docs."}

# Log incoming requests to help debug preflight/headers
@app.post("/generate_pitch")
async def generate_pitch(request: Request, data: PitchRequest):
    # Debug: print method & relevant headers for preflight debugging
    print(">>> Request method:", request.method)
    headers = dict(request.headers)
    # Print only the most useful headers to avoid spam
    print(">>> Origin:", headers.get("origin"))
    print(">>> Access-Control-Request-Method:", headers.get("access-control-request-method"))
    print(">>> Content-Type:", headers.get("content-type"))

    prompt = (
        f"Generate a detailed 200+ word startup pitch for the following:\n\n"
        f"Idea: {data.idea}\n"
        f"Industry: {data.industry}\n"
        f"Tone: {data.tone}\n"
        f"Target Audience: {data.audience}\n\n"
        f"The pitch should clearly outline the problem, solution, product features, and potential impact."
    )

    headers_out = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8
    }

    try:
        resp = requests.post(API_URL, headers=headers_out, json=payload)
        print("Groq status:", resp.status_code)
        if resp.status_code != 200:
            print("Groq response text:", resp.text)
            return {"error": "Groq returned non-200", "details": resp.text}

        res_data = resp.json()
        if "choices" not in res_data or not res_data["choices"]:
            return {"error": "Unexpected response from Groq", "details": res_data}

        pitch_text = res_data["choices"][0]["message"]["content"]
        return {"pitch": pitch_text}

    except Exception as e:
        print("Exception while calling Groq:", e)
        return {"error": "Server error", "details": str(e)}
