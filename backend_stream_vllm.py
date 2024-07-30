# backend.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
from fastapi.responses import StreamingResponse
from vllm import LLM, SamplingParams
import asyncio

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model using vLLM
model = LLM(model="Qwen/Qwen-7B-Chat-Int4", trust_remote_code=True)

class ChatRequest(BaseModel):
    message: str
    history: list = []

async def generate_response(message: str):
    sampling_params = SamplingParams(temperature=0.7, max_tokens=1000)
    async for output in model.stream_complete(message, sampling_params):
        cleaned_text = output.text.strip()
        if cleaned_text:
            #print(cleaned_text)
            yield cleaned_text + "\n"

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    message = chat_request.message

    # Debugging logs
    logging.info(f"Received message: {message}")

    try:
        return StreamingResponse(generate_response(message), media_type="text/plain")
    except Exception as e:
        logging.error(f"Error during generation: {str(e)}")
        return {"response": "An error occurred during generation."}

# Run the backend in a separate thread
def run_backend():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_backend()
