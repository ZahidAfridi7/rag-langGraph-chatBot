from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from vector_store import load_and_embed
from rag_graph import rag_chatbot
from langchain_core.messages import HumanMessage

import subprocess, threading, os

app = FastAPI(title="Hybrid RAG: OpenAI Embeddings + Groq LLM")

@app.on_event("startup")
async def startup_event():
    def run_ui():
        subprocess.run(["streamlit", "run", "frontend.py", "--server.port", "8501"])
    threading.Thread(target=run_ui, daemon=True).start()
    print("⚡ Streamlit running at http://localhost:8501")


@app.post("/upload")
async def upload_file(file: UploadFile):
    path = f"./data/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    load_and_embed(path)
    return {"status": "File embedded successfully!"}


@app.post("/chat")
async def chat(query: str = Form(...), thread_id: str = Form("1")):
    config = {"configurable": {"thread_id": thread_id}}

    result = rag_chatbot.invoke(
        {"question": query, "messages": [HumanMessage(content=query)]},
        config=config
    )
    return {"answer": result["messages"][-1].content}
