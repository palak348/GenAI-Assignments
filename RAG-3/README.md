# DocChat — Document-Grounded RAG Application

A Retrieval-Augmented Generation (RAG) application that lets a user upload a document (PDF, TXT, or Markdown) and ask natural-language questions about its contents. Answers are generated strictly from retrieved passages of the uploaded document, with page-level citations.

> Submitted for **Assignment 03 — Google NotebookLM RAG**.

---

## Overview

The system implements a complete RAG pipeline:

1. **Ingestion** — text is extracted from the uploaded document.
2. **Chunking** — text is split into semantically coherent passages.
3. **Embedding** — each passage is converted to a 768-dimensional vector.
4. **Storage** — vectors and metadata are upserted into a vector database.
5. **Retrieval** — at query time, the top-k most similar passages are fetched.
6. **Generation** — an LLM produces an answer constrained to the retrieved context.

The user interacts with the system through a chat interface; every answer is accompanied by the supporting passages, their source filenames, page numbers, and similarity scores.

---

## Tech Stack

| Layer            | Technology                                       |
| ---------------- | ------------------------------------------------ |
| Application      | Python 3.11+, Streamlit                          |
| LLM              | Google Gemini `gemini-2.5-flash-lite`            |
| Embeddings       | Google Gemini `gemini-embedding-001` (768-dim)   |
| Vector Database  | Qdrant Cloud (cosine similarity)                 |
| PDF Parsing      | `pypdf`                                          |
| Text Chunking    | `langchain-text-splitters`                       |
| Deployment       | Streamlit Community Cloud                        |

---

## Architecture

```
┌────────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Document  │ → │  Chunk   │ → │   Embed    │ → │  Qdrant  │ → │ Retrieve │ → │  Gemini  │
│ PDF/TXT/MD │   │ Recursive│   │  Gemini    │   │  Vector  │   │  top-k   │   │  Answer  │
│            │   │ Splitter │   │ embedding  │   │   Store  │   │ similar  │   │ grounded │
│            │   │ 1000/150 │   │   001      │   │  cosine  │   │  chunks  │   │  in docs │
└────────────┘   └──────────┘   └────────────┘   └──────────┘   └──────────┘   └──────────┘
```

---

## Chunking Strategy

The application uses `RecursiveCharacterTextSplitter` configured as follows:

| Parameter        | Value                                  |
| ---------------- | -------------------------------------- |
| `chunk_size`     | `1000` characters                      |
| `chunk_overlap`  | `150` characters                       |
| `separators`     | `["\n\n", "\n", ". ", " ", ""]`        |

The splitter attempts to split on the largest semantic boundary first (paragraph, then line, then sentence, then word, then character). This preserves coherent units of meaning within each chunk and avoids cutting mid-sentence. The 150-character overlap ensures that information sitting on a chunk boundary remains retrievable from at least one of the two adjacent chunks.

Chunks are produced per page so that the page number is preserved as metadata and surfaced in the final answer as `[p. N]` citations.

---

## Answer Grounding

The generation prompt explicitly constrains the model to use only the retrieved context:

- The system prompt forbids the use of outside knowledge.
- If the retrieved context does not contain the answer, the model is instructed to reply *"I couldn't find that in the document."*
- Generation runs at `temperature = 0.2` to minimise speculation.
- The model receives only the top-5 retrieved chunks — never the full document or any external information.

The retrieved chunks are also displayed to the user in an expandable "Sources" panel so that grounding can be verified.

---

## Local Setup

### 1. Prerequisites

| Service        | Where to obtain                          |
| -------------- | ---------------------------------------- |
| Gemini API key | https://aistudio.google.com/apikey       |
| Qdrant cluster | https://cloud.qdrant.io                  |

### 2. Install

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

### 3. Configure environment

Copy the template and fill in the three values:

```bash
cp .env.example .env
```

```dotenv
GEMINI_API_KEY=...
QDRANT_URL=https://<cluster-id>.<region>.cloud.qdrant.io
QDRANT_API_KEY=...
```

### 4. Run

```bash
streamlit run app.py
```

The application is served at `http://localhost:8501`.

---

## Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub.
2. Sign in at https://share.streamlit.io.
3. Create a new app pointing to this repository, with main file path `RAG-3/app.py`.
4. Under **Advanced settings → Secrets**, provide:

   ```toml
   GEMINI_API_KEY = "..."
   QDRANT_URL = "..."
   QDRANT_API_KEY = "..."
   ```

5. Deploy.

---

## Project Structure

```
RAG-3/
├── app.py                       # Streamlit UI and chat orchestration
├── rag.py                       # RAG pipeline (chunk, embed, store, retrieve, generate)
├── requirements.txt             # Python dependencies
├── .env.example                 # Local environment template
├── .gitignore
├── .streamlit/
│   ├── config.toml              # UI theme and upload settings
│   └── secrets.toml.example     # Streamlit secrets template
└── README.md
```

---

## Author

**Palak Agrawal**
