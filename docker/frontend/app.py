import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="LocalRAG Chat", page_icon="🤖", layout="centered")
st.title("🤖 LocalRAG Chat")
st.caption(f"Connected to backend: `{BACKEND_URL}`")


def check_backend() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def send_message(message: str) -> str:
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={"message": message},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["reply"]
    except requests.exceptions.ConnectionError:
        return "Error: Cannot reach backend. Is it running?"
    except Exception as e:
        return f"Error: {e}"


def load_history() -> list[dict]:
    try:
        r = requests.get(f"{BACKEND_URL}/api/history", timeout=5)
        r.raise_for_status()
        return r.json().get("history", [])
    except Exception:
        return []


# Sidebar
with st.sidebar:
    st.header("Status")
    if check_backend():
        st.success("Backend connected")
    else:
        st.error("Backend unreachable")

    if st.button("Clear History"):
        try:
            requests.delete(f"{BACKEND_URL}/api/history", timeout=5)
            st.session_state.pop("chat_history", None)
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")

# Load history into session on first run
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history()

# Render chat messages
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(msg["user"])
    with st.chat_message("assistant"):
        st.write(msg["bot"])

# Input
if prompt := st.chat_input("Type your message…"):
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking…"):
        reply = send_message(prompt)

    with st.chat_message("assistant"):
        st.write(reply)

    st.session_state.chat_history.append({"user": prompt, "bot": reply})
