import os
import json
import uuid
import tempfile
from typing import Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Agentic RAG Dashboard",
    page_icon=":brain:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "steps" not in st.session_state:
    st.session_state.steps = []
if "web_search_used" not in st.session_state:
    st.session_state.web_search_used = False

# ---------------------------------------------------------------------------
# Sidebar: agent trace + document upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Agent Trace")
    if st.session_state.steps:
        for i, step in enumerate(st.session_state.steps, 1):
            node_label = step.get("node", "unknown")
            detail = step.get("detail", "")
            with st.expander(f"{i}. {node_label}", expanded=False):
                st.caption(detail)
        if st.session_state.web_search_used:
            st.info("Web search fallback was triggered.")
    else:
        st.caption("Send a message to see the agent trace here.")

    st.divider()
    st.header("Upload Documents")
    uploaded_file = st.file_uploader(
        "Add a PDF or TXT file to the vector store",
        type=["pdf", "txt"],
    )
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        if st.button("Ingest into ChromaDB"):
            with st.spinner("Uploading..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/api/upload",
                        params={"file_path": tmp_path},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"Uploaded {data['filename']} – "
                            f"{data['chunks_added']} chunk(s) added."
                        )
                    else:
                        st.error(resp.json().get("detail", "Upload failed"))
                except requests.ConnectionError:
                    st.error("Cannot reach the backend. Is the API running?")
                finally:
                    os.unlink(tmp_path)

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:12]}…`")
    if st.button("New Session"):
        st.session_state.messages = []
        st.session_state.steps = []
        st.session_state.web_search_used = False
        st.session_state.session_id = uuid.uuid4().hex
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Agentic RAG Dashboard")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Handle user input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking…")

        try:
            resp = requests.post(
                f"{BACKEND_URL}/api/chat",
                json={
                    "prompt": prompt,
                    "session_id": st.session_state.session_id,
                },
                stream=True,
                timeout=180,
            )

            answer_chunks: list[str] = []
            steps: list[dict] = []
            web_search_used = False

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if "error" in obj:
                    placeholder.error(f"Backend error: {obj['error']}")
                    break
                if "answer" in obj:
                    answer_chunks.append(obj["answer"])
                    steps = obj.get("steps", [])
                    web_search_used = obj.get("web_search_used", False)
                elif "node" in obj:
                    steps.append(obj)

            if answer_chunks:
                full_answer = answer_chunks[-1]
                placeholder.markdown(full_answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_answer}
                )
                st.session_state.steps = steps
                st.session_state.web_search_used = web_search_used
            else:
                placeholder.markdown("No response received.")

        except requests.ConnectionError:
            placeholder.error(
                "Cannot connect to the backend API. "
                "Make sure the FastAPI server is running on "
                f"`{BACKEND_URL}`."
            )
        except Exception as exc:
            placeholder.error(f"Unexpected error: {exc}")
