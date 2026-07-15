"""
Streamlit frontend -- a simple web chat UI that calls the FastAPI backend.

Why Streamlit instead of building a custom React app: Streamlit turns a
plain Python script into a working web UI with almost no frontend code
(no HTML/CSS/JS needed) -- ideal for a personal project where the backend
(the actual RAG pipeline) is the interesting part to demonstrate, not
frontend polish. A production product would use something like React
instead, but Streamlit is the right tool for "show this working in a
browser" without weeks of frontend work.

Run with:
    streamlit run frontend/app.py

This expects the FastAPI backend to already be running separately at
http://localhost:8000 (run `uvicorn api.main:app --reload` in another
terminal first).
"""
import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Personal RAG Chatbot", page_icon="🤖")
st.title("🤖 Personal RAG Chatbot")
st.caption("Ask about my resume, background, or the LLM/RAG papers in my knowledge base.")

# A stable session_id per browser session, so the backend keeps separate
# conversation memory for each user/tab instead of mixing them together.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...} for display only

# Redraw the whole conversation on every rerun (Streamlit's normal pattern)
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
                response = requests.post(
                    API_URL,
                    json={"session_id": st.session_state.session_id, "message": user_input},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]
            except requests.exceptions.ConnectionError:
                answer = (
                    "Could not reach the backend API. Make sure it's running: "
                    "`uvicorn api.main:app --reload --port 8000` in a separate terminal."
                )
                sources = []
            except Exception as e:
                answer = f"Something went wrong: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
