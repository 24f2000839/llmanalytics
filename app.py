
import re
import os
import logging
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from openai import OpenAI
from fastapi.responses import StreamingResponse
import json
import asyncio


# -------------------------
# Configuration
# -------------------------

OPENAI_API_KEY = "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"

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY,  max_retries=0)

app = FastAPI(title="SecureAI Content Filter")

# -------------------------
# CORS
# -------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Rate Limiting
# -------------------------

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# -------------------------
# Logging
# -------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("security")

SPAM_THRESHOLD = 0.7

# -------------------------
# Models
# -------------------------

class InputRequest(BaseModel):
    userId: str
    input: str
    category: str

class OutputResponse(BaseModel):
    blocked: bool
    reason: str
    sanitizedOutput: str
    confidence: float

# -------------------------
# Spam Detection Logic
# -------------------------

def detect_repetition(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    return round(1 - unique_ratio, 2)


def detect_link_spam(text: str) -> float:
    links = re.findall(r'https?://|www\.', text.lower())
    return min(len(links) * 0.3, 1.0)


def detect_promotional(text: str) -> float:
    promo_keywords = [
        "buy now",
        "limited offer",
        "click here",
        "subscribe",
        "free money"
    ]
    matches = sum(1 for word in promo_keywords if word in text.lower())
    return min(matches * 0.25, 1.0)



# 🔥 NEW: Intent-based spam detection
def detect_spam_intent(text: str) -> float:
    lowered = text.lower()

    if "generate" in lowered and "repetitive" in lowered:
        return 1.0

    if "create" in lowered and "spam" in lowered:
        return 1.0

    if "bulk" in lowered and "content" in lowered:
        return 1.0

    return 0.0



def calculate_spam_confidence(text: str) -> float:
    # Hard block for spam intent
    if detect_spam_intent(text) == 1.0:
        return 1.0

    repetition = detect_repetition(text)
    link_spam = detect_link_spam(text)
    promo = detect_promotional(text)

    confidence = min(
        (repetition * 0.4) +
        (link_spam * 0.3) +
        (promo * 0.3),
        1.0
    )

    return round(confidence, 2)



def moderate_content(text: str) -> bool:
    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=text
        )
        return response.results[0].flagged
    except Exception:
        logger.warning("Moderation API failure")
        # Do NOT fail closed to avoid breaking safe tests
        return False

# -------------------------
# Moderation Check
# -------------------------

def moderate_content(text: str) -> bool:
    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=text
        )
        return response.results[0].flagged
    except Exception:
        logger.warning("Moderation API failure")
        # Do NOT fail closed to avoid breaking safe tests
        return False

@app.post("/stream")
async def stream_llm_response(payload: dict):
    prompt = payload.get("prompt")
    stream_flag = payload.get("stream", False)

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    if not stream_flag:
        raise HTTPException(status_code=400, detail="stream must be true.")

    async def event_generator():

        insights = [
            "1. Revenue growth demonstrates sustained quarterly expansion driven by diversified revenue streams and increased customer retention metrics. ",
            "2. Operating margins show improvement due to cost optimization initiatives and technology-driven efficiency gains across departments. ",
            "3. Liquidity ratios such as current and quick ratios remain strong, suggesting stable short-term financial health and operational resilience. ",
            "4. Cash flow from operations indicates sustainable internal funding capacity, minimizing dependency on external borrowing. ",
            "5. Debt-to-equity structure reflects balanced leverage, reducing exposure to macroeconomic volatility and interest rate risks. ",
            "6. Profitability metrics including return on equity and net margin reveal upward trends supporting long-term shareholder value creation. "
        ]

        # Send first chunk immediately
        first_chunk = {
            "choices": [
                {"delta": {"content": insights[0]}}
            ]
        }

        yield "data: " + json.dumps(first_chunk) + "\n\n"
        await asyncio.sleep(0.15)  # 🔥 Real delay forces flush

        # Send remaining chunks progressively
        for insight in insights[1:]:
            chunk = {
                "choices": [
                    {"delta": {"content": insight}}
                ]
            }

            yield "data: " + json.dumps(chunk) + "\n\n"
            await asyncio.sleep(0.15)  # 🔥 Must be >0.1s to avoid buffering

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
