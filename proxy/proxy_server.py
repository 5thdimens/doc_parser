import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded



limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)



VLLM_URL = os.getenv("VLLM_URL")


def filter_input(text: str) -> str:
    if "some_forbidden_word" in text.lower():
        raise HTTPException(status_code=400, detail="Inappropriate content detected.")
    return text

def filter_output(text: str) -> str:
    return text.replace("secret_data", "[REDACTED]")


@app.post("/v1/chat/completions")
@limiter.limit("1/minute")
async def proxy_handler(request: Request):
    if not VLLM_URL:
        raise HTTPException(status_code=500, detail="VLLM_URL environment variable is not set.")

    payload = await request.json()

    # Filter the user's message
    if 'messages' in payload:
        for message in payload['messages']:
            if message.get('role') == 'user':
                message['content'] = filter_input(message['content'])

    # Forward the request to vLLM
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(VLLM_URL, json=payload)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Error forwarding request to model: {e}")

    vllm_data = response.json()

    # Filter the model's response
    if vllm_data.get("choices"):
        content = vllm_data["choices"][0]["message"]["content"]
        vllm_data["choices"][0]["message"]["content"] = filter_output(content)

    return vllm_data



@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

