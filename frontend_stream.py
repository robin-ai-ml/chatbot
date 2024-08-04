# frontend.py

import httpx
import gradio as gr
import requests
from datetime import datetime

def get_current_time():
    current_timestamp = datetime.now()
    formated_time = current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return formated_time

# Define the chat function to interact with the backend
def chat_fn(message, chatbot):
    print(f"Message to backend: {message}, {get_current_time()}")  # Debugging output
    try:
        with httpx.stream("GET", "http://localhost:8000/chat", json={"message": message, "history": []}, timeout=None) as response:
            full_response = ""
            for chunk in response.iter_text():
                if chunk:
                    full_response += chunk
                    #print(f'recv next token {chunk.strip()}: {get_current_time()}')
                    chatbot[-1][1] = full_response.strip()
                    yield chatbot

    except httpx.RequestError as e:
        chatbot.append([message, f"Error: {str(e)}"])
        print(f"Appended to chatbot (error): {chatbot[-1]}")  # Debugging output
        print(f"Chatbot content (error): {chatbot}")  # Debugging output
        yield chatbot



with gr.Blocks() as demo:
    with gr.Row():
        gr.Image("callcenter1.jpg", elem_id="logo", show_label=False)
        gr.Markdown("<h1>Call Center</h1>")
    
    chatbot = gr.Chatbot()
    message = gr.Textbox(show_label=False, placeholder="Type your message here...")
    clear = gr.Button("Clear")

    # The user function now returns an empty string to clear the textbox and keeps the message
    def user(message, chatbot):
        
        print(f"User input: {message}, {get_current_time()}")  # Debugging output
        chatbot.append([message, ""])
        return "", chatbot, message

    # Update the message submit method to clear the input box and show the conversation
    message.submit(user, [message, chatbot], [message, chatbot, message], queue=False).then(
        chat_fn, [message, chatbot], [chatbot], queue=True
    )
    
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch(share=True)
