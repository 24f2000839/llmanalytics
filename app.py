import re
import logging
from typing import Dict
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from openai import OpenAI

# -------------------------
# Configuration
# -------------------------

OPENAI_API_KEY = "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"

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="SecureAI Content Filter")

# Rate limiting: 5 requests per minute per IP
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

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
    repetition_score = 1 - unique_ratio
    return repetition_score


def detect_link_spam(text: str) -> float:
    links = re.findall(r'https?://|www\.', text.lower())
    return min(len(links) * 0.3, 1.0)


def detect_promotional(text: str) -> float:
    promo_keywords = ["buy now", "limited offer", "click here", "subscribe", "free money"]
    matches = sum(1 for word in promo_keywords if word in text.lower())
    return min(matches * 0.25, 1.0)


def calculate_spam_confidence(text: str) -> float:
    repetition = detect_repetition(text)
    link_spam = detect_link_spam(text)
    promo = detect_promotional(text)

    # Weighted average
    confidence = min((repetition * 0.4) + (link_spam * 0.3) + (promo * 0.3), 1.0)
    return round(confidence, 2)


# -------------------------
# Moderation Check
# -------------------------

def moderate_content(text: str) -> bool:
    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=text
        )
        flagged = response.results[0].flagged
        return flagged
    except Exception:
        logger.warning("Moderation API failure")
        return False


# -------------------------
# Endpoint
# -------------------------

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Too many requests. Please try again later."
    )


@app.post("/validate", response_model=OutputResponse)
@limiter.limit("5/minute")
async def validate_input(request: Request, payload: InputRequest):

    if not payload.userId or not payload.input:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input."
        )

    if payload.category != "Content Filtering":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid category."
        )

    try:
        spam_confidence = calculate_spam_confidence(payload.input)
        moderation_flag = moderate_content(payload.input)

        if spam_confidence > SPAM_THRESHOLD or moderation_flag:
            logger.warning(f"Blocked content from user {payload.userId}")
            return OutputResponse(
                blocked=True,
                reason="Spam or policy violation detected",
                sanitizedOutput="",
                confidence=spam_confidence
            )

        sanitized = payload.input.strip()

        return OutputResponse(
            blocked=False,
            reason="Input passed all security checks",
            sanitizedOutput=sanitized,
            confidence=1 - spam_confidence
        )

    except Exception:
        logger.error("Internal validation error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Request could not be processed."
        )
