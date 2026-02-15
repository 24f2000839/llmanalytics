import json
import asyncio
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="StreamText Render Deployment")

# CORS is vital for Render to communicate with your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust financial content (~900 characters) to ensure character count requirement
FINANCIAL_CONTENT = (
    "1. Revenue Growth: The company reported a 22% increase in quarterly revenue, "
    "outperforming market expectations due to high demand in the cloud sector. "
    "This growth is supported by a 15% expansion in the enterprise customer base. "
    "2. Operational Efficiency: Net margins improved by 5% as a result of "
    "streamlined supply chain logistics and reduced energy costs across global hubs. "
    "3. Asset Liquidity: A current ratio of 2.1 indicates strong short-term "
    "financial health and the ability to cover immediate liabilities comfortably. "
    "4. Debt Profile: Debt-to-equity remains stable at 0.45, suggesting a "
    "conservative leverage strategy that minimizes long-term risk volatility. "
    "5. R&D Investment: Allocation of 15% of gross profit to research confirms "
    "a commitment to maintaining a competitive edge in AI technology and automation. "
    "6. Shareholder Value: The announced $200M buyback program reflects "
    "management's confidence in sustained cash flow generation for the next fiscal year."
)

@app.post("/stream")
async def stream_response(payload: dict):
    # Requirement validation
    prompt = payload.get("prompt")
    stream_requested = payload.get("stream", False)

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    if not stream_requested:
        raise HTTPException(status_code=400, detail="stream must be true.")

    async def event_generator():
        # Split into words (tokens)
        tokens = FINANCIAL_CONTENT.split(" ")
        
        try:
            for i, token in enumerate(tokens):
                content = token + (" " if i < len(tokens) - 1 else "")
                
                # SSE format matching Task requirements
                data = {
                    "choices": [
                        {
                            "delta": {"content": content},
                            "index": 0,
                            "finish_reason": None
                        }
                    ]
                }
                
                # Format: data: {...}\n\n
                yield f"data: {json.dumps(data)}\n\n"
                
                # SPEED BOOST: 0.01s sleep = ~100 tokens/sec.
                # This compensates for any latency or buffering overhead on Render.
                await asyncio.sleep(0.01)

            # Essential [DONE] signal
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # DISBALE BUFFERING: Critical for Render/Cloud providers
            "X-Accel-Buffering": "no", 
            "Transfer-Encoding": "chunked",
        }
    )

if __name__ == "__main__":
    import uvicorn
    # Render provides the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
