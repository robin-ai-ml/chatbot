
# frontend.py

import gradio as gr
import requests

# Define the chat function to interact with the backend
def chat_fn(message, chatbot):
    #print(f"Message to backend: {message}")  # Debugging output
    try:
        response = requests.post("http://localhost:8000/chat", json={"message": message, "history": []})
        response.raise_for_status()
        response_json = response.json()
        # Append the new message and response to the chatbot
        chatbot.append([message, response_json["response"]])
        #print(f"Appended to chatbot: {chatbot[-1]}")  # Debugging output
        #print(f"Chatbot content: {chatbot}")  # Debugging output
        return chatbot
    except requests.exceptions.RequestException as e:
        chatbot.append([message, f"Error: {str(e)}"])
        print(f"Appended to chatbot (error): {chatbot[-1]}")  # Debugging output
        print(f"Chatbot content (error): {chatbot}")  # Debugging output
        return chatbot

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
        return "", chatbot, message

    # Update the message submit method to clear the input box and show the conversation
    message.submit(user, [message, chatbot], [message, chatbot, message], queue=False).then(
        chat_fn, [message, chatbot], [chatbot], queue=False
    ).then(
        lambda chatbot: (chatbot, ""), [chatbot], [chatbot, message], queue=False
    )
    
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch(share=True)



