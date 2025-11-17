
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader

DATA_DIR = "./data"
DB_DIR = "./chroma_db"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# OpenAI embeddings (cheap + high quality)
embeddings = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL"))

# Chroma vector DB
chroma = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

def load_and_embed(filepath):
    """Embed PDF, DOCX, or CSV using OpenAI embeddings."""
    
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    elif filepath.endswith(".csv"):
        loader = CSVLoader(filepath)
    else:
        raise Exception("Unsupported file type")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    chroma.add_documents(chunks)

def get_retriever():
    return chroma.as_retriever(search_kwargs={"k": 3})
