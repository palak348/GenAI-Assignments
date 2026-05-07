# QueryLeaf

> Upload a document. Ask questions. Get answers grounded in the source — never hallucinated.

A production-style Retrieval-Augmented Generation (RAG) application that turns any uploaded document into an interactive, searchable knowledge base. Every answer is constrained to the document's own contents and accompanied by the exact passages used to construct it, with page-level citations.

**🔗 Live demo:** https://queryleaf.streamlit.app

![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4.svg)
![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-DC382D.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Pipeline in Detail](#pipeline-in-detail)
  - [1. Ingestion](#1-ingestion)
  - [2. Chunking](#2-chunking)
  - [3. Embedding](#3-embedding)
  - [4. Storage](#4-storage)
  - [5. Retrieval](#5-retrieval)
  - [6. Generation](#6-generation)
- [Answer Grounding](#answer-grounding)
- [Tech Stack](#tech-stack)
- [Local Setup](#local-setup)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [Limitations and Future Work](#limitations-and-future-work)
- [License](#license)

---

## Overview

A typical session looks like this:

```
1. User uploads "research-paper.pdf"
   → 12 pages, 38 chunks indexed in 4 seconds.

2. User: "What dataset did the authors evaluate on?"
   → "The authors evaluated their model on the MS-COCO 2017
      validation split, using the standard 5K image subset. [p. 4]"

   Sources panel:
     Excerpt 1 — research-paper.pdf · p. 4 · score 0.821
     > We evaluate our model on the MS-COCO 2017 validation split...
     Excerpt 2 — research-paper.pdf · p. 4 · score 0.778
     > ...the standard 5K subset, following prior work in this area...

3. User: "What is the capital of France?"
   → "I couldn't find that in the document."
```

The third interaction is the important one: the model is forced to refuse questions that cannot be answered from the uploaded source.

---

## Features

- **Multi-format ingestion** — PDF, plain text, and Markdown.
- **Robust PDF cleaning** — repairs the spurious newlines that `pypdf` emits on layouts with code blocks or narrow columns, so chunks contain coherent prose rather than fragmented words.
- **Semantic chunking** — recursive splitting on paragraph → line → sentence boundaries, with overlap to protect facts that fall on chunk edges.
- **Vector retrieval** — cosine-similarity search over Gemini embeddings stored in Qdrant Cloud.
- **Strict grounding** — the LLM is constrained to answer only from retrieved passages, with explicit page citations.
- **Source transparency** — every answer ships with an expandable "Sources" panel showing the exact chunks, their page numbers, and their similarity scores.
- **Per-session isolation** — each user session indexes into its own collection, so simultaneous users do not contaminate each other's documents.

---

## Architecture

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Document │ → │  Clean   │ → │  Embed   │ → │  Qdrant  │ → │ Retrieve │ → │  Gemini  │
│ PDF/TXT  │   │   and    │   │  Gemini  │   │  Vector  │   │  top-k   │   │  Answer  │
│   /MD    │   │  Chunk   │   │ embedding│   │  Store   │   │ similar  │   │ grounded │
│          │   │          │   │   001    │   │  cosine  │   │  chunks  │   │  in docs │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                                                                 │
                                                                                 ▼
                                                                       ┌────────────────────┐
                                                                       │  Answer + cited    │
                                                                       │  source passages   │
                                                                       └────────────────────┘
```

The user-facing surface is a Streamlit chat interface. On every turn the user receives both the model's answer and the supporting passages, so grounding can be independently verified.

---

## Pipeline in Detail

### 1. Ingestion

| Format | Library  | Behaviour                                                       |
| ------ | -------- | --------------------------------------------------------------- |
| PDF    | `pypdf`  | Page-by-page text extraction; each page passes through a custom cleaner. |
| TXT    | built-in | Decoded as UTF-8.                                               |
| MD     | built-in | Decoded as UTF-8 (Markdown syntax is treated as plain prose).   |

The PDF cleaner addresses a well-known `pypdf` limitation: depending on a document's layout (code blocks, narrow columns, non-standard fonts), the extractor inserts one or more newlines between every word, producing one-word-per-line output. The cleaner therefore:

1. Normalises line endings.
2. Stitches hyphen-broken words back together (`exam-\nple` → `example`).
3. Collapses every run of whitespace into a single space.

With clean prose in hand, the downstream chunker can rely on **sentence boundaries** — the most semantically meaningful unit — rather than `pypdf`'s unreliable whitespace signal.

### 2. Chunking

Implemented with `langchain_text_splitters.RecursiveCharacterTextSplitter`:

| Parameter        | Value                              |
| ---------------- | ---------------------------------- |
| `chunk_size`     | `1000` characters                  |
| `chunk_overlap`  | `150` characters                   |
| `separators`     | `["\n\n", "\n", ". ", " ", ""]`    |

**Why recursive splitting?** A naive fixed-size splitter cuts mid-sentence and frequently mid-word. The recursive splitter tries the largest semantic boundary first (paragraph), and only descends to smaller units (line, sentence, word, character) when a chunk would otherwise exceed `chunk_size`. The result is chunks that respect natural meaning units.

**Why 1000 / 150?** A chunk of ~1000 characters typically captures three to six full sentences, which gives the embedding model enough context to produce a meaningful vector while staying small enough that retrieval can pinpoint the relevant region of a long document. 150 characters of overlap keeps a fact intact when it sits on a chunk boundary.

**Why per-page chunking?** Chunks are produced page by page so each one inherits its source page number. That metadata propagates through retrieval and surfaces in the final answer as `[p. N]` citations.

### 3. Embedding

| Setting       | Value                                                            |
| ------------- | ---------------------------------------------------------------- |
| Model         | `models/gemini-embedding-001`                                    |
| Output dim    | `768`                                                            |
| Task type     | `RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for queries   |
| Batch size    | 100 chunks per API call                                          |

Gemini's `gemini-embedding-001` supports asymmetric task types: documents and queries are embedded under different objectives even though they share a model. This noticeably improves retrieval quality compared to embedding both sides identically.

### 4. Storage

Vectors and metadata are upserted into a **Qdrant Cloud** collection scoped to the user's session id. Each point carries:

- The chunk text.
- The source filename.
- The originating page number.

Cosine similarity is the distance metric. A new collection is created on first upload; the user can reset their session at any time, which deletes the collection.

### 5. Retrieval

For each query:

1. The query is embedded with the same model (using `RETRIEVAL_QUERY` task type).
2. The top **k = 5** nearest neighbours are fetched from Qdrant.
3. Each result carries its similarity score, used both for ranking and for display in the Sources panel.

### 6. Generation

| Setting              | Value                          |
| -------------------- | ------------------------------ |
| Model                | `gemini-2.5-flash-lite`        |
| Temperature          | `0.2`                          |
| Max output tokens    | `1024`                         |
| System prompt        | Strict grounding (see below)   |

The retrieved chunks are formatted into a numbered context block with their page numbers and source filenames, then passed to Gemini together with the user's question.

---

## Answer Grounding

Grounding is enforced at three layers:

1. **System prompt.** The model is explicitly instructed to answer only from the supplied context, to refuse with *"I couldn't find that in the document."* when the context is insufficient, and to attach `[p. N]` citations.
2. **Low temperature.** Generation runs at `temperature = 0.2`, which strongly discourages speculation while still allowing natural phrasing.
3. **Bounded context.** The model is given only the top-5 retrieved chunks — never the full document, never the open web, never any prior conversation that wasn't itself grounded.

The retrieved chunks are then surfaced to the user in an expandable Sources panel, so grounding can be independently verified rather than taken on faith.

---

## Tech Stack

| Layer            | Technology                                         |
| ---------------- | -------------------------------------------------- |
| Application      | Python 3.11+, Streamlit                            |
| LLM              | Google Gemini `gemini-2.5-flash-lite`              |
| Embeddings       | Google Gemini `gemini-embedding-001` (768-dim)     |
| Vector Database  | Qdrant Cloud, cosine similarity                    |
| PDF Parsing      | `pypdf`                                            |
| Text Chunking    | `langchain-text-splitters`                         |
| Deployment       | Streamlit Community Cloud                          |

---

## Local Setup

### Prerequisites

| Service        | Where to obtain                          |
| -------------- | ---------------------------------------- |
| Gemini API key | https://aistudio.google.com/apikey       |
| Qdrant cluster | https://cloud.qdrant.io                  |

### Installation

```bash
git clone https://github.com/palak348/GenAI-Assignments.git
cd GenAI-Assignments/RAG-3

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Copy the template and fill in the three values:

```bash
cp .env.example .env
```

```dotenv
GEMINI_API_KEY=...
QDRANT_URL=https://<cluster-id>.<region>.cloud.qdrant.io
QDRANT_API_KEY=...
```

### Run

```bash
streamlit run app.py
```

The application is served at `http://localhost:8501`.

---

## Deployment

The application is deployed on **Streamlit Community Cloud**:

1. Push the repository to GitHub.
2. Sign in at https://share.streamlit.io.
3. Create a new app pointing at this repository, with main file path `RAG-3/app.py`.
4. In **Advanced settings → Python version**, select **3.11**.
5. In the **Secrets** field, provide the three keys in TOML form:

   ```toml
   GEMINI_API_KEY = "..."
   QDRANT_URL = "..."
   QDRANT_API_KEY = "..."
   ```

6. Click **Deploy**.

Subsequent pushes to `main` trigger an automatic redeploy.

---

## Project Structure

```
RAG-3/
├── app.py                       # Streamlit UI and chat orchestration
├── rag.py                       # RAG pipeline (load → clean → chunk → embed → store → retrieve → generate)
├── requirements.txt             # Pinned Python dependencies
├── .env.example                 # Local environment template
├── .gitignore
├── .streamlit/
│   ├── config.toml              # UI theme and upload size
│   └── secrets.toml.example     # Streamlit Cloud secrets template
└── README.md
```

The codebase is split into two modules so that the RAG pipeline (`rag.py`) is fully decoupled from the UI (`app.py`). The pipeline functions accept explicit clients and configuration and return plain data, which makes them straightforward to test, reuse from a CLI, or wrap in a different frontend.

---

## Design Decisions

A few choices that are not visible in the code itself but materially affect quality:

- **Streamlit over a custom REST + React split.** A single-file UI with native chat, file upload, and session state was the right shape for this scope. Splitting into a backend and a frontend would have doubled the surface area without improving the user experience.
- **Cloud-hosted vector database over an in-process one.** Qdrant Cloud was preferred over an embedded store (Chroma, FAISS) because the deployment target's filesystem is ephemeral; an in-process index would be lost on every redeploy.
- **API-based embeddings over local `sentence-transformers`.** Local embeddings would avoid third-party rate limits, but would also pull 500 MB+ of PyTorch and model weights into the deploy image, which strains free-tier hosts. Gemini embeddings keep the deployment lightweight.
- **`gemini-2.5-flash-lite` over `gemini-2.0-flash`.** Each model has its own free-tier quota bucket, so picking a less-trafficked model gives more headroom; `2.5-flash-lite` also benefits from architectural improvements over the 2.0 family.
- **Aggressive whitespace flattening on PDF text.** `pypdf`'s whitespace output is unreliable enough that any heuristic for paragraph detection is wrong on at least some documents. Flattening to single spaces and letting the chunker split on sentence boundaries is more robust across diverse PDF layouts.

---

## Limitations and Future Work

What this implementation does *not* yet do:

- **No multi-document chat.** Each session is scoped to one document at a time. Supporting multi-document Q&A would require either combining collections at query time or attaching document filters to retrieval.
- **No conversational memory across turns.** Each question is answered in isolation; the model does not see the previous turn. Extending this would require careful design, since naive history injection can break grounding.
- **No reranking.** Retrieval is pure cosine similarity over Gemini embeddings. A cross-encoder reranker over the top-20 results would likely improve precision on ambiguous queries.
- **No OCR.** Scanned-image PDFs without an embedded text layer are not readable; an OCR step (e.g. Tesseract) would be needed to support those.
- **No streaming responses.** Answers are returned in one shot rather than streamed token by token. Streamlit supports streaming, and switching the Gemini call to a streaming generator would feel snappier on long answers.

---

## License

Released under the MIT License.
