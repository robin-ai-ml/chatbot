# frontend.py

import gradio as gr
import requests

# Define the chat function to interact with the backend
def chat_fn(message, chatbot):
    print(f"Message to backend: {message}")  # Debugging output
    try:
        response = requests.post("http://localhost:8000/chat", json={"message": message, "history": []}, stream=True)
        full_response = ""
        for chunk in response.iter_content(decode_unicode=True):
            full_response += chunk
            chatbot[-1][1] = full_response.strip()
            yield chatbot
    except requests.exceptions.RequestException as e:
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
        print(f"User input: {message}")  # Debugging output
        chatbot.append([message, ""])
        return "", chatbot, message

    # Update the message submit method to clear the input box and show the conversation
    message.submit(user, [message, chatbot], [message, chatbot, message], queue=False).then(
        chat_fn, [message, chatbot], [chatbot], queue=True
    )
    
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch(share=True)
