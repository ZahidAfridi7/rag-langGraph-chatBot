import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag_graph import rag_chatbot, checkpointer
import uuid

# ======================================================
# Utility Functions
# ======================================================

def generate_thread_id():
    return str(uuid.uuid4())

def retrieve_all_threads():
    threads = []
    for cp in checkpointer.list(None):
        threads.append(cp.config["configurable"]["thread_id"])
    return threads

def load_conversation(thread_id):
    state = rag_chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    add_thread(new_id)
    st.session_state["message_history"] = []


# ======================================================
# Session Setup
# ======================================================
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

add_thread(st.session_state["thread_id"])


# ======================================================
# Sidebar
# ======================================================
st.sidebar.title("💬 LangGraph RAG Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.subheader("📁 My Conversations")

for tid in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(tid), key=f"thread_button_{tid}"):
        st.session_state["thread_id"] = tid
        msgs = load_conversation(tid)

        rendered = []
        for msg in msgs:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            rendered.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = rendered


# ======================================================
# Main UI (Chat Bubbles + Streaming)
# ======================================================
st.title("🤖 AI Assistant (Groq + LangGraph + RAG)")

# Render old messages
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Render user bubble
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Stream Groq response with chat bubble
    with st.chat_message("assistant"):
        full_response = ""

        def token_stream():
            for chunk, meta in rag_chatbot.stream(
                {
                    "question": user_input,
                    "messages": [HumanMessage(content=user_input)],
                },
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessage):
                    text = chunk.content
                    full_response_list.append(text)   # store it
                    yield text

        # Use list to capture tokens without nonlocal
        full_response_list = []

        st.write_stream(token_stream())

        # Merge all streamed tokens
        full_response = "".join(full_response_list)

    # Save AI bubble
    st.session_state["message_history"].append(
        {"role": "assistant", "content": full_response}
    )
