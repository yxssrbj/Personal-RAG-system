import os
from pathlib import Path
import tempfile
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
import sqlite3

MESSAGES_DIR = Path(__file__).resolve().parent.parent / "db"
MESSAGES_DIR.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(MESSAGES_DIR / "messages.db")
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, role, content)")
cur.execute(
    "CREATE TABLE IF NOT EXISTS conversations("
    "id INTEGER PRIMARY KEY, title TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
)
try:
    cur.execute("ALTER TABLE messages ADD COLUMN conversation_id INTEGER")
except sqlite3.OperationalError:
    pass
con.commit()

def save_message(role, content):
    conversation_id = st.session_state.conversation_id
    st.session_state.messages.append({"role": role, "content": content})
    cur.execute("INSERT INTO messages (role, content, conversation_id) VALUES (?, ?, ?)", (role, content, conversation_id))
    con.commit()


def start_new_conversation(title="New chat"):
    cur.execute('INSERT INTO conversations (title) VALUES (?)', (title,))
    con.commit()
    conversation_id = cur.lastrowid
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = []

load_dotenv()

os.environ["USER_AGENT"] = "PersonalDocQA/1.0"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.com"

st.set_page_config(page_title="Personal Document Q&A", page_icon="📄", layout="centered")
st.title("📄 Personal Document Q&A")
st.caption("Upload PDF documents and chat with them in a simple Streamlit interface.")

PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "personal_documents"


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



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

if "conversation_id" not in st.session_state:
    start_new_conversation()

# if "messages" not in st.session_state:
#     st.session_state.messages = []
#     res = cur.execute('SELECT role, content FROM messages ORDER BY id')
#     for msg in res.fetchall():
#         st.session_state.messages.append({"role":msg[0], "content":msg[1]})

    
## the logic
# -> start with empty messages list on startup and new conversation row in conversations table
# -> give me a temporary palce holder for new chats

if "retriever" not in st.session_state:
    existing_store = Chroma(
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    if existing_store._collection.count() > 0:
        st.session_state.retriever = existing_store.as_retriever(search_kwargs={"k": 4})
    else:
        st.session_state.retriever = None

if "conversation_id" not in st.session_state:
    pass

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_retriever(uploaded_files):

    temp_files = []
    try:
        documents = []
        for uploaded_file in uploaded_files:
            original_name = uploaded_file.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            temp_files.append(temp_path)
            loader = PyPDFLoader(temp_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata['source'] = original_name
            documents.extend(loaded_docs)


        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=80)
        splits = splitter.split_documents(documents)

        vectorestore = Chroma.from_documents(
            splits,
            get_embeddings(),
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME)
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
    st.header("Model")
    selected_model = st.selectbox("Choose a model", options=["Ollama 3.2"])
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Choose PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Process PDFs") :
        if uploaded_files:
            with st.spinner("Loading and indexing your PDFs..."):
                st.session_state.retriever = build_retriever(uploaded_files)
                
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
    
    save_message("user",prompt)
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
            with st.chat_message("assistant"):
                sources = []

                def answer_tokens():
                    for chunk in rag_chain.stream({"input": prompt, "chat_history": chat_history}):
                        if "context" in chunk:
                            sources.extend(chunk["context"])
                        if "answer" in chunk:
                            yield chunk["answer"]

                answer = st.write_stream(answer_tokens())

                if answer:
                    with st.expander("Sources"):
                        for doc in sources:
                            page = doc.metadata.get("page", "?")
                            source = Path(doc.metadata.get("source", "unknown")).name
                            st.caption(f"{source} - page {page + 1 if isinstance(page, int) else page}")
        except Exception as e:
            st.error(
                "Couldn't reach the Ollama model. Make sure Ollama is running "
                "(`ollama serve`) and that the model is pulled (`ollama pull llama3.2`)."
            )
            st.caption(f"Details: {e}")

    if answer:
        save_message("assistant",answer)
