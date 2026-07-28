from langchain_core.messages import HumanMessage, AIMessage
import requests


def check_ollama_heath(model_name="llama3.2", base_url="http://localhost:11434"):
    try:
        response = requests.get(base_url + "/api/tags", timeout=3)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return False, "Ollama doesn't seem to be running"
    
    models = [m['name'] for m in response.json().get("models",[])]
    if not any(m == model_name or m.startswith(f"{model_name}") for m in models):
        return False, f"Model '{model_name}' isn't pulled yet."
    return True, "Ollama is ready"

def build_chat_history(messages):
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

