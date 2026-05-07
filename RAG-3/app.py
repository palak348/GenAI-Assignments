"""
Streamlit UI for the NotebookLM-style RAG app.

Run locally:
    streamlit run app.py

Required secrets / env vars:
    GEMINI_API_KEY       (required)
    QDRANT_URL           (required, e.g. https://xyz.qdrant.io:6333)
    QDRANT_API_KEY       (required for Qdrant Cloud, optional for local)
"""

from __future__ import annotations

import hashlib
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

import rag

load_dotenv()

st.set_page_config(
    page_title="DocChat — your own NotebookLM",
    page_icon="📚",
    layout="wide",
)


# ---------- Config helpers ----------

def get_secret(key: str) -> str | None:
    """Read from st.secrets first (Streamlit Cloud), then environment."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except (FileNotFoundError, KeyError):
        pass
    return os.environ.get(key)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
QDRANT_URL = get_secret("QDRANT_URL")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY")


# ---------- Session bootstrap ----------

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []  # list of {filename, pages, chunks}


def collection_name() -> str:
    return f"docchat_{st.session_state.session_id}"


# ---------- Sidebar ----------

with st.sidebar:
    st.title("📚 DocChat")
    st.caption("Your own Google NotebookLM, free and open.")

    config_ok = bool(GEMINI_API_KEY and QDRANT_URL)
    if not config_ok:
        st.error(
            "Missing config. Set **GEMINI_API_KEY** and **QDRANT_URL** "
            "(and **QDRANT_API_KEY** if using Qdrant Cloud) in `.env` or "
            "Streamlit secrets."
        )
    else:
        st.success("Connected to Gemini + Qdrant.")

    st.divider()
    st.subheader("1. Upload a document")
    uploaded = st.file_uploader(
        "PDF, TXT, or MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False,
        disabled=not config_ok,
    )

    if uploaded and config_ok:
        file_bytes = uploaded.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        already_indexed = any(d.get("hash") == file_hash for d in st.session_state.indexed_docs)

        if already_indexed:
            st.info(f"`{uploaded.name}` is already indexed in this session.")
        else:
            with st.status(f"Indexing `{uploaded.name}`...", expanded=False) as status:
                try:
                    rag.configure_gemini(GEMINI_API_KEY)
                    client = rag.get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)

                    st.write("Loading & extracting text...")
                    st.write("Chunking with RecursiveCharacterTextSplitter (1000 chars, 150 overlap)...")
                    st.write("Embedding with Gemini `gemini-embedding-001` (768d)...")
                    st.write("Storing vectors in Qdrant...")

                    info = rag.index_document(
                        client=client,
                        collection=collection_name(),
                        filename=uploaded.name,
                        file_bytes=file_bytes,
                    )
                    info["hash"] = file_hash
                    st.session_state.indexed_docs.append(info)
                    status.update(
                        label=f"✅ Indexed `{uploaded.name}` — {info['pages']} pages, {info['chunks']} chunks",
                        state="complete",
                    )
                except Exception as e:
                    status.update(label=f"❌ Indexing failed", state="error")
                    st.exception(e)

    st.divider()
    st.subheader("Indexed documents")
    if not st.session_state.indexed_docs:
        st.caption("No documents yet — upload one above.")
    else:
        for d in st.session_state.indexed_docs:
            st.markdown(f"- **{d['filename']}** · {d['pages']} pages · {d['chunks']} chunks")

    if st.session_state.indexed_docs and st.button("🗑️ Clear this session", use_container_width=True):
        try:
            client = rag.get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
            if client.collection_exists(collection_name()):
                client.delete_collection(collection_name())
        except Exception as e:
            st.warning(f"Couldn't delete collection: {e}")
        st.session_state.indexed_docs = []
        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.rerun()

    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown(
            """
            **Pipeline:**
            1. **Ingest** — extract text from PDF / TXT / MD
            2. **Chunk** — RecursiveCharacterTextSplitter (1000 chars, 150 overlap)
            3. **Embed** — Gemini `text-embedding-004` (768-dim)
            4. **Store** — Qdrant vector DB (cosine similarity)
            5. **Retrieve** — top-k similar chunks for the query
            6. **Generate** — Gemini `gemini-2.0-flash` answers using ONLY retrieved chunks
            """
        )


# ---------- Main chat area ----------

st.title("Chat with your document")

if not st.session_state.indexed_docs:
    st.info("👈 Upload a document in the sidebar to get started.")
else:
    docs_label = ", ".join(f"`{d['filename']}`" for d in st.session_state.indexed_docs)
    st.caption(f"Asking about: {docs_label}")

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], start=1):
                    page = f"p. {src['page']}" if src.get("page") else "n/a"
                    st.markdown(f"**Excerpt {i}** — `{src['source']}` · {page} · score `{src['score']:.3f}`")
                    st.markdown(f"> {src['text']}")

# Chat input
prompt = st.chat_input(
    "Ask a question about your document...",
    disabled=not (st.session_state.indexed_docs and GEMINI_API_KEY and QDRANT_URL),
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            try:
                rag.configure_gemini(GEMINI_API_KEY)
                client = rag.get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
                answer, chunks = rag.answer_query(
                    client=client,
                    collection=collection_name(),
                    query=prompt,
                    k=5,
                )
            except Exception as e:
                st.exception(e)
                answer, chunks = "Sorry — something went wrong.", []

        st.markdown(answer)

        sources = [
            {"text": c.text, "page": c.page, "score": c.score, "source": c.source}
            for c in chunks
        ]
        if sources:
            with st.expander(f"📎 Sources ({len(sources)} chunks)"):
                for i, src in enumerate(sources, start=1):
                    page = f"p. {src['page']}" if src.get("page") else "n/a"
                    st.markdown(f"**Excerpt {i}** — `{src['source']}` · {page} · score `{src['score']:.3f}`")
                    st.markdown(f"> {src['text']}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
