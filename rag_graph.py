
from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_groq import ChatGroq

from vector_store import get_retriever
from prompt import SYSTEM_PROMPT

# LLM from Groq
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("LLM_MODEL"),
    temperature=0.2,
)

retriever = get_retriever()

class ChatState(dict):
    question: str
    context: str
    messages: list[BaseMessage]

# 1. Retrieve context from Chroma
def retrieve_node(state: ChatState):
    question = state["question"]
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    return {"context": context}

# 2. Generate response using Groq LLM
def rag_node(state: ChatState):
    prompt = f"""
{SYSTEM_PROMPT}

Context:
{state['context']}

Question:
{state['question']}
"""
    answer = llm.invoke(prompt)
    return {"messages": [answer]}

# Build LangGraph
graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", rag_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# SQLite thread-based memory
conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

rag_chatbot = graph.compile(checkpointer=checkpointer)
