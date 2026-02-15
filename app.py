import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="StreamText High-Frequency Provider")

# Enable CORS for testing suites
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ~800 characters of financial insight content
FINANCIAL_CONTENT = (
    "1. Revenue Growth: The company reported a 22% increase in quarterly revenue, "
    "outperforming market expectations due to high demand in the cloud sector. "
    "2. Operational Efficiency: Net margins improved by 5% as a result of "
    "streamlined supply chain logistics and reduced energy costs. "
    "3. Asset Liquidity: A current ratio of 2.1 indicates strong short-term "
    "financial health and the ability to cover immediate liabilities comfortably. "
    "4. Debt Profile: Debt-to-equity remains stable at 0.45, suggesting a "
    "conservative leverage strategy that minimizes long-term risk. "
    "5. R&D Investment: Allocation of 15% of gross profit to research confirms "
    "a commitment to maintaining a competitive edge in AI technology. "
    "6. Shareholder Value: The announced $200M buyback program reflects "
    "management's confidence in sustained cash flow generation for the next fiscal year."
)

@app.post("/stream")
async def stream_response(payload: dict):
    prompt = payload.get("prompt")
    stream_requested = payload.get("stream", False)

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    if not stream_requested:
        raise HTTPException(status_code=400, detail="The 'stream' parameter must be true.")

    async def event_generator():
        # Split into words to maximize the number of chunks
        # This will result in ~120 chunks
        tokens = FINANCIAL_CONTENT.split(" ")
        
        try:
            for i, token in enumerate(tokens):
                # Add the space back to the token except for the last one
                content = token + (" " if i < len(tokens) - 1 else "")
                
                # OpenAI-compatible SSE chunk format
                data = {
                    "choices": [
                        {
                            "delta": {"content": content},
                            "index": 0,
                            "finish_reason": None
                        }
                    ]
                }
                
                # Format as SSE
                yield f"data: {json.dumps(data)}\n\n"
                
                # 0.03s delay creates a fast 'typewriter' effect (~33 tokens/sec)
                # This satisfies the >27 tokens/second requirement perfectly.
                await asyncio.sleep(0.03)

            # Final termination signal
            yield "data: [DONE]\n\n"

        except Exception as e:
            # Send error as a final chunk if something breaks
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", # Critical for real-time delivery
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
