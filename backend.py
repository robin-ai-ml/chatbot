# backend.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
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

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    message = chat_request.message
    history = chat_request.history if chat_request.history is not None else []

    # Debugging logs
    #logging.info(f"Received message: {message}")
    #logging.info(f"Received history: {history}")

    # Generate the response using the model
    try:
        response, history = model.chat(tokenizer, message, history=history)
        #logging.info(f"Response: {response}")
        return {"response": response, "history": history}
    except Exception as e:
        #logging.error(f"Error during generation: {str(e)}")
        return {"response": "An error occurred during generation.", "history": history}

# Run the backend in a separate thread
def run_backend():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_backend()



