from langchain_core.messages import HumanMessage, AIMessage


def build_chat_history(messages):
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)