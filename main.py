import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
import groq as groq_errors
from openai import OpenAI
import openai as openai_errors

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # optional — enables a non-Groq fallback

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
    that reasoning models occasionally leak even with reasoning hidden."""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Fallback chain. Each entry is tried in order until one returns real content.
# The first three all live on Groq but are separate models -> separate quota
# buckets, so hitting a rate limit on one doesn't block the others.
# The last one (optional) is a completely different provider/account, so it
# survives a full Groq outage or exhausting every Groq model's daily limit.
# ---------------------------------------------------------------------------

def call_groq(messages, model, reasoning_effort=None, reasoning_format=None):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    client = Groq(api_key=GROQ_API_KEY)
    kwargs = {}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if reasoning_format is not None:
        kwargs["reasoning_format"] = reasoning_format

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
        **kwargs,
    )
    return completion.choices[0].message.content


def call_openrouter(messages, model):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )
    return completion.choices[0].message.content


# (label, function, kwargs) — order = priority
FALLBACK_CHAIN = [
    ("groq: qwen3.6-27b (non-thinking)", call_groq,
        {"model": "qwen/qwen3.6-27b", "reasoning_effort": "none"}),
    ("groq: qwen3.6-27b (thinking, hidden)", call_groq,
        {"model": "qwen/qwen3.6-27b", "reasoning_effort": "default", "reasoning_format": "hidden"}),
    ("groq: gpt-oss-120b", call_groq,
        {"model": "openai/gpt-oss-120b", "reasoning_effort": "low"}),
    ("groq: gpt-oss-20b", call_groq,
        {"model": "openai/gpt-oss-20b", "reasoning_effort": "low"}),
    # Free-tier OpenRouter model as a last resort, only used if OPENROUTER_API_KEY is set.
    # Swap the model string for whichever free OpenRouter model is current when you deploy.
    ("openrouter: fallback model", call_openrouter,
        {"model": "meta-llama/llama-3.3-70b-instruct:free"}),
]

RETRYABLE_ERRORS = (
    groq_errors.RateLimitError,
    groq_errors.APIStatusError,
    openai_errors.RateLimitError,
    openai_errors.APIStatusError,
)


@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    if not payload.history:
        return {"response": "No message received."}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": m.role, "content": m.content} for m in payload.history
    ]

    last_error = None

    for label, fn, kwargs in FALLBACK_CHAIN:
        try:
            reply = fn(messages, **kwargs)
            reply = strip_thinking_tags(reply)
            if reply:
                return {"response": reply}
            # Empty content (e.g. reasoning ate the whole token budget) -> try next in chain
            print(f"[{label}] returned empty content, falling back")
            last_error = "empty content"
        except RETRYABLE_ERRORS as e:
            print(f"[{label}] rate-limited or API error, falling back: {e}")
            last_error = str(e)
            continue
        except RuntimeError as e:
            # Missing API key for that provider — skip silently, not a real failure
            continue
        except Exception as e:
            print(f"[{label}] unexpected error, falling back: {e}")
            last_error = str(e)
            continue

    return {
        "response": (
            "I'm having trouble reaching my AI engine right now (all providers are "
            "rate-limited or unavailable). Please try again in a minute."
        )
    }


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(
            content=html,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return "<h1>Frontend index.html file not found in directory root</h1>"
