import os
import re
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


def strip_thinking_tags(text: str) -> str:
    """Defensive safety net: removes any stray <think>...</think> blocks
    that reasoning models occasionally leak even with reasoning_format='hidden'."""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    if not API_KEY:
        return {"response": "Backend Server Error: The GROQ_API_KEY environment variable is blank or missing inside Render settings."}

    if not payload.history:
        return {"response": "No message received."}

    try:
        client = Groq(api_key=API_KEY)

        # Send the FULL conversation history (not just the last message) so the
        # model retains context across turns — required for the Mock Interview flow.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            {"role": m.role, "content": m.content} for m in payload.history
        ]

        completion = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            reasoning_format="hidden",  # hides internal chain-of-thought; only final answer returned
        )

        reply = completion.choices[0].message.content
        reply = strip_thinking_tags(reply)

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
