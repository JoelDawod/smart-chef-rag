import requests
import gradio as gr

# Service configuration
BACKEND_URL = "http://127.0.0.1:8000/ask-chef"

def chat_with_chef(user_message: str, chat_history: list) -> str:
    """
    Transmits user queries to the FastAPI backend and retrieves the generated response.
    Handles basic connection errors and HTTP status checks.
    """
    try:
        payload = {"query": user_message}
        response = requests.post(BACKEND_URL, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return data["answer"]
        else:
            return f"Backend Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "Connection Error: Could not connect to the backend server. Ensure FastAPI is running."

# Initialize the UI components
demo = gr.ChatInterface(
    fn=chat_with_chef,
    title="👨‍🍳 الشيف العربي الذكي",
    description="أخبرني بالمكونات المتوفرة لديك، وسأقترح عليك وصفات رائعة لتحضيرها!"
)

if __name__ == "__main__":
    print("Launching Gradio UI server...")
    demo.launch()