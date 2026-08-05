import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = (
    "You are an expert AI Career Guidance Coach. Your goal is to provide actionable direction, "
    "resume feedback, and interview practice. Adhere to these rules strictly: 1. Be encouraging, "
    "realistic, and highly practical. 2. If the user asks for a Career Roadmap, break it down into "
    "phases (Skills to learn, Projects to build, Where to apply). 3. If the user asks for a Resume "
    "Review, highlight specific bullet points and recommend strong action verbs. 4. If the user "
    "wants a Mock Interview, ask ONLY ONE interview question at a time. Wait for their response, "
    "provide immediate constructive feedback on their answer, and then ask the next question."
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    history: list[ChatMessage]

@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server configuration error: Missing API Key.")
    
    try:
        # Initializing the model with system prompts directly
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT
        )
        
        # Formulate structured payload match for official Gemini endpoints
        formatted_contents = []
        for msg in payload.history:
            # Match internal Gemini roles ('user' or 'model')
            target_role = "user" if msg.role == "user" else "model"
            formatted_contents.append({
                "role": target_role,
                "parts": [{"text": msg.content}]
            })
            
        # Fast direct generation bypasses chat tracking errors
        response = model.generate_content(contents=formatted_contents)
        
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    # Looks explicitly for index.html in the same directory
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend index.html file not found in directory root</h1>"
