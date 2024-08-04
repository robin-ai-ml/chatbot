from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging
from fastapi.responses import StreamingResponse
from datetime import datetime
import time
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import json
from transformers import AutoTokenizer

def get_current_time():
    current_timestamp = datetime.now()
    formatted_time = current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return formatted_time

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

# Initialize vLLM engine without quantization
engine_args = AsyncEngineArgs(
    model="Qwen/Qwen-7B-Chat-Int4",
    tensor_parallel_size=1,
    enforce_eager=True,
    trust_remote_code=True,
    quantization=None  # Ensure quantization is disabled
)
engine = AsyncLLMEngine.from_engine_args(engine_args)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat-Int4", trust_remote_code=True)

class ChatRequest(BaseModel):
    message: str
    history: list = []

async def generate_response(message: str):
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95,  max_tokens=4096)
    request_id = time.monotonic()
    previous_text = ""
    results_generator = engine.generate(message, sampling_params, request_id=request_id)

    async for request_output in results_generator:
        text_outputs = [output.text for output in request_output.outputs]
        text_output = text_outputs[0][len(previous_text):].strip()
        cleaned_text =  tokenizer.decode(tokenizer.encode(text_output), skip_special_tokens=True).strip()
        if cleaned_text:
            #logging.info(f"yield next token: {cleaned_text} {get_current_time()}")
            previous_text = text_outputs[0]
            yield f"{cleaned_text} "

@app.get("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    message = chat_request.message

    # Debugging logs

    logging.info(f"Received message: {message}, {get_current_time()}")

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