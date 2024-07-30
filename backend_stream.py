# backend.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import logging
from threading import Thread
from fastapi.responses import StreamingResponse
import torch

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

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat-Int4", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat-Int4", device_map="auto", trust_remote_code=True).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class ChatRequest(BaseModel):
    message: str
    history: list = []

async def generate_response(message: str):
    inputs = tokenizer([message], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer)
    
    # Run the generation in a separate thread
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1000)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        cleaned_text = tokenizer.decode(tokenizer.encode(new_text), skip_special_tokens=True).strip()
        if cleaned_text:
            yield f"{cleaned_text} "

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    message = chat_request.message

    # Debugging logs
    logging.info(f"Received message: {message}")

    try:
        return  StreamingResponse(generate_response(message), media_type="text/event-stream")

    except Exception as e:
        logging.error(f"Error during generation: {str(e)}")
        return {"response": "An error occurred during generation."}

# Run the backend in a separate thread
def run_backend():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_backend()
