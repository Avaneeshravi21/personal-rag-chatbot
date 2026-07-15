"""
Standalone deployment version for HuggingFace Spaces (Streamlit SDK).

Why this is different from frontend/app.py: locally, we run a separate
FastAPI backend + Streamlit frontend as two processes, which mirrors a
real production architecture (useful to demonstrate that skill). But
HuggingFace's free Streamlit Space hosting runs ONE process -- so for
deployment, this file imports the RAGPipeline directly instead of
making HTTP calls to a separate server. Same underlying pipeline code
either way (core/pipeline.py) -- just wired together differently for
this single-process hosting environment.

This file must be named app.py and sit at the PROJECT ROOT for
HuggingFace's Streamlit SDK to find and run it automatically.
"""
import streamlit as st
import uuid
from core.pipeline import RAGPipeline

st.set_page_config(page_title="Personal RAG Chatbot", page_icon="🤖")
st.title("🤖 Personal RAG Chatbot")
st.caption("Ask about my background, or the LLM/RAG research papers in my knowledge base.")

# Loaded ONCE per Space instance (cached across all users/reruns), not
# once per browser session -- loading the embedding model and reranker
# takes real time, so every visitor sharing one loaded pipeline instance
# keeps the app responsive instead of reloading models per visitor.
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                result = pipeline.ask(st.session_state.session_id, user_input)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = f"Something went wrong: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
