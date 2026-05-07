"""
RAG pipeline: ingestion -> chunking -> embedding -> storage -> retrieval -> generation.

Chunking strategy: RecursiveCharacterTextSplitter
  - chunk_size = 1000 characters
  - chunk_overlap = 150 characters
  - Splits on paragraph -> line -> sentence -> word boundaries (in that order),
    preserving semantic units. Overlap keeps cross-chunk context so answers
    aren't cut off at boundaries.
"""

from __future__ import annotations

import io
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Iterable

import google.generativeai as genai
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768
GENERATION_MODEL = "gemini-2.5-flash-lite"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH = 100


@dataclass
class RetrievedChunk:
    text: str
    page: int | None
    score: float
    source: str


def configure_gemini(api_key: str) -> None:
    genai.configure(api_key=api_key)


def get_qdrant_client(url: str, api_key: str | None) -> QdrantClient:
    if api_key:
        return QdrantClient(url=url, api_key=api_key, prefer_grpc=False, timeout=60)
    return QdrantClient(url=url, prefer_grpc=False, timeout=60)


def ensure_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=EMBEDDING_DIM,
            distance=qmodels.Distance.COSINE,
        ),
    )


def reset_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    ensure_collection(client, name)


# ---------- Ingestion ----------

def clean_pdf_text(text: str) -> str:
    """Repair pypdf output where each visual line ends in a newline.

    pypdf's extract_text() inserts a newline between every text fragment,
    so PDFs with narrow columns or short lines come out as one-word-per-line.
    This collapses single newlines into spaces while preserving paragraph
    breaks (>= 2 newlines) and stitching hyphen-broken words back together.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=\w)", "", text)
    placeholder = "<<<PARAGRAPH_BREAK>>>"
    text = re.sub(r"\n{2,}", placeholder, text)
    text = re.sub(r"\n+", " ", text)
    text = text.replace(placeholder, "\n\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def load_pdf(file_bytes: bytes) -> list[tuple[int, str]]:
    """Return list of (page_number, page_text). Page numbers are 1-indexed."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean_pdf_text(raw)
        if text.strip():
            pages.append((i, text))
    return pages


def load_text(file_bytes: bytes) -> list[tuple[int, str]]:
    text = file_bytes.decode("utf-8", errors="ignore")
    return [(1, text)] if text.strip() else []


def load_document(filename: str, file_bytes: bytes) -> list[tuple[int, str]]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return load_pdf(file_bytes)
    if lower.endswith(".txt") or lower.endswith(".md"):
        return load_text(file_bytes)
    raise ValueError(f"Unsupported file type: {filename}. Use PDF, TXT, or MD.")


# ---------- Chunking ----------

def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks: list[dict] = []
    for page_num, page_text in pages:
        for piece in splitter.split_text(page_text):
            piece = piece.strip()
            if piece:
                chunks.append({"text": piece, "page": page_num})
    return chunks


# ---------- Embedding ----------

def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed texts in batches. task_type is RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=batch,
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        )
        embeddings = result["embedding"]
        if isinstance(embeddings[0], float):
            vectors.append(embeddings)
        else:
            vectors.extend(embeddings)
    return vectors


# ---------- Indexing ----------

def index_document(
    client: QdrantClient,
    collection: str,
    filename: str,
    file_bytes: bytes,
) -> dict:
    pages = load_document(filename, file_bytes)
    if not pages:
        raise ValueError("No extractable text found in the document.")

    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("Chunking produced no content.")

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "text": chunk["text"],
                "page": chunk["page"],
                "source": filename,
            },
        )
        for chunk, vec in zip(chunks, vectors)
    ]

    ensure_collection(client, collection)
    batch = 128
    for i in range(0, len(points), batch):
        client.upsert(collection_name=collection, points=points[i:i + batch], wait=True)

    return {
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
    }


# ---------- Retrieval ----------

def retrieve(
    client: QdrantClient,
    collection: str,
    query: str,
    k: int = 5,
) -> list[RetrievedChunk]:
    if not client.collection_exists(collection):
        return []
    query_vec = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    hits = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=k,
        with_payload=True,
    ).points
    out: list[RetrievedChunk] = []
    for h in hits:
        payload = h.payload or {}
        out.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                page=payload.get("page"),
                score=float(h.score) if h.score is not None else 0.0,
                source=payload.get("source", ""),
            )
        )
    return out


# ---------- Generation ----------

SYSTEM_PROMPT = """You are a document-grounded assistant. Answer the user's question using ONLY the context excerpts below, which come from a document the user uploaded.

Strict rules:
- If the answer is not in the context, say: "I couldn't find that in the document."
- Do NOT use outside knowledge. Do NOT speculate.
- Cite the page number(s) you used in square brackets, e.g. [p. 4].
- Keep the answer focused and concise. Use bullet points if it improves clarity.
"""


def build_context_block(chunks: Iterable[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        page = f"p. {c.page}" if c.page else "n/a"
        parts.append(f"[Excerpt {i} | {page} | source: {c.source}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "I couldn't find that in the document. Please upload a document first or try rephrasing."

    context = build_context_block(chunks)
    model = genai.GenerativeModel(
        model_name=GENERATION_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    return (response.text or "").strip() or "I couldn't find that in the document."


def answer_query(
    client: QdrantClient,
    collection: str,
    query: str,
    k: int = 5,
) -> tuple[str, list[RetrievedChunk]]:
    chunks = retrieve(client, collection, query, k=k)
    answer = generate_answer(query, chunks)
    return answer, chunks
