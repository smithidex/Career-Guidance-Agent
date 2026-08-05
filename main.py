import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pull key safely from Render environment settings
API_KEY = os.getenv("GROQ_API_KEY")

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
        return {"response": "Backend Server Error: The GROQ_API_KEY environment variable is blank or missing inside Render settings."}
    
    try:
        # Initialize Groq client securely
        client = Groq(api_key=API_KEY)
        
        # Extract the latest text user input
        user_message = payload.history[-1].content
        
        # Call Groq's high-speed Llama 3 70B model
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        
        reply = completion.choices[0].message.content
        if reply:
            return {"response": reply}
        else:
            return {"response": "API Warning: Received empty text from Groq engine."}
            
    except Exception as e:
        error_msg = str(e)
        print(f"CRITICAL ENGINE ERROR: {error_msg}")
        return {"response": f"Groq Engine Error Details: {error_msg}"}

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend index.html file not found in directory root</h1>"

