import os
from pathlib import Path
import tempfile
import uuid
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import requests
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()

os.environ["USER_AGENT"] = "PersonalDocQA/1.0"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.com"

st.set_page_config(page_title="Personal Document Q&A", page_icon="📄", layout="centered")
st.title("📄 Personal Document Q&A")
st.caption("Upload PDF documents and chat with them in a simple Streamlit interface.")



def build_chat_history(messages):
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


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

ollama_ok, ollama_message = check_ollama_heath()

if not ollama_ok:
    st.error(ollama_message)
    st.stop()


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_retriever(uploaded_files, collection_name):

    temp_files = []
    try:
        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            temp_files.append(temp_path)
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=80)
        splits = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        persist_dir = str(Path(__file__).resolve().parent.parent / "chroma_db")
        vectorestore = Chroma.from_documents(
            splits,
            embeddings,
            persist_directory=persist_dir,
            collection_name=collection_name)
        return vectorestore.as_retriever(search_kwargs={"k": 4})
    finally:
        for temp_path in temp_files:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass


contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question which might reference "
     "context in the chat history, formulate a standalone question that can be "
     "understood without the chat history. Do NOT answer the question — just "
     "reformulate it if needed, and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
llm = ChatOllama(model="llama3.2")

with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Choose PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Process PDFs") :
        if uploaded_files:
            with st.spinner("Loading and indexing your PDFs..."):
                st.session_state.retriever = build_retriever(uploaded_files, st.session_state.session_id)
                
                st.success("Documents processed successfully.")
        else:
            st.warning("Please upload at least one PDF first.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your uploaded documents")

if prompt:
    if st.session_state.retriever is None:
        st.info("Upload and process a PDF before asking questions.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, contextualize_prompt
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
    ("system",
     "Answer the question based only on the following context:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ]
    )
    

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    chat_history = build_chat_history(st.session_state.messages[:-1])
    answer = None
    with st.spinner("Thinking..."):
        try:
            result = rag_chain.invoke({
                "input":prompt,
                "chat_history":chat_history
            })
            answer = result['answer']
        except Exception as e:
            st.error(
            "Couldn't reach the Ollama model. Make sure Ollama is running "
            "(`ollama serve`) and that the model is pulled (`ollama pull llama3.2`)."
        )
            st.caption(f"Details: {e}")
    if answer is not None:
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
