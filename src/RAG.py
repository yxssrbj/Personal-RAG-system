import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

os.environ["USER_AGENT"] = "PersonalDocQA/1.0"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.langchain.com'
\
## uploading the materials

loader = WebBaseLoader(
    web_path=('https://lilianweng.github.io/posts/2023-06-23-agent/'),
    bs_kwargs=dict(
        parse_only= bs4.SoupStrainer(
            class_=('post-title', 'post-content', 'post-header'))

    )
)

docs = loader.load()

## splitting

spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50
)

splits = spliter.split_documents(docs)

## embedding in vectors

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorestore = Chroma.from_documents(splits, embeddings)
retriever = vectorestore.as_retriever()


prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}
""")

llm = ChatOllama(model="llama3.2")

## getting useful chunks

question = input('Ask your question')

chunks = retriever.invoke(question)

## using the chunks + question to generate a response

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


result = rag_chain.invoke(question)
print(result)