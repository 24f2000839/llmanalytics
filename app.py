import re
import os
import logging
import json
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from openai import OpenAI

# -------------------------
# Configuration
# -------------------------
# IMPORTANT: Use environment variables. Never hardcode keys in production.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-key-here")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="SecureAI Content Filter")

# -------------------------
# Middleware & Rate Limiting
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("security")

# -------------------------
# Helper Logic
# -------------------------
def moderate_content(text: str) -> bool:
    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=text
        )
        return response.results[0].flagged
    except Exception as e:
        logger.warning(f"Moderation API failure: {e}")
        return False

# -------------------------
# Streaming Endpoint
# -------------------------
@app.post("/stream")
async def stream_llm_response(payload: dict):
    prompt = payload.get("prompt")
    stream_flag = payload.get("stream", False)

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    if not stream_flag:
        # If they don't want a stream, we shouldn't use StreamingResponse
        raise HTTPException(status_code=400, detail="Set 'stream': true for this endpoint.")

    async def event_generator():
        insights = [
            "1. Revenue growth demonstrates sustained quarterly expansion. ",
            "2. Operating margins show improvement due to cost optimization. ",
            "3. Liquidity ratios remain strong, suggesting stable health. ",
            "4. Cash flow indicates sustainable internal funding capacity. ",
            "5. Debt-to-equity structure reflects balanced leverage. ",
            "6. Profitability metrics reveal upward trends for shareholders. "
        ]

        try:
            for insight in insights:
                # 1. Create the OpenAI-style chunk
                chunk = {
                    "choices": [
                        {"delta": {"content": insight}, "index": 0, "finish_reason": None}
                    ]
                }
                
                # 2. Format as Server-Sent Event (SSE)
                # We yield BYTES to prevent FastAPI from trying to buffer strings
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                
                # 3. Small sleep to ensure the event loop flushes the buffer
                await asyncio.sleep(0.1)

            # 4. Final signal
            yield b"data: [DONE]\n\n"

        except asyncio.CancelledError:
            logger.info("Client disconnected from stream")
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables Nginx buffering
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
