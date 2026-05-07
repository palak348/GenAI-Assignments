# 📚 DocChat — Your Own Google NotebookLM

A fully open-source, **100% free** RAG-powered app where you upload any document (PDF / TXT / MD) and chat with it. Answers are grounded in the document only — no hallucinations from the LLM's general knowledge.

> Assignment 03 — Google NotebookLM RAG

---

## ✨ Features

- 📄 Upload PDF, TXT, or MD files
- 🧠 Chunked, embedded, and indexed in a real vector database
- 💬 Chat UI with conversation history
- 📎 Every answer shows the exact source chunks (with page numbers and similarity scores)
- 🎯 Grounded answers — the model is forced to use ONLY the retrieved chunks
- 🆓 Zero paid services, no credit card required anywhere

---

## 🏗️ Architecture

```
┌─────────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐
│  Document   │ → │  Chunk   │ → │  Embed   │ → │ Qdrant  │ → │ Retrieve │ → │  Gemini  │
│ (PDF/TXT)   │   │ Recursive│   │  Gemini  │   │ vector  │   │  top-k   │   │  Flash   │
│             │   │  splitter│   │  embed-  │   │   DB    │   │          │   │  answer  │
│             │   │ 1000/150 │   │  ding-004│   │         │   │          │   │          │
└─────────────┘   └──────────┘   └──────────┘   └─────────┘   └──────────┘   └──────────┘
```

### Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Best free RAG ecosystem |
| UI | Streamlit | Free Streamlit Community Cloud hosting |
| LLM | Google Gemini 2.5 Flash Lite | Free tier (no credit card), 1M token/day |
| Embeddings | Gemini `gemini-embedding-001` (768-d) | Same free key as the LLM |
| Vector DB | Qdrant Cloud (free 1 GB cluster) | Persistent storage, no credit card |
| PDF parsing | `pypdf` | Pure Python, free |
| Chunking | `langchain-text-splitters` | Battle-tested recursive splitter |

---

## 🧩 Chunking Strategy

We use **`RecursiveCharacterTextSplitter`** with these parameters:

- **`chunk_size = 1000`** characters
- **`chunk_overlap = 150`** characters
- **`separators = ["\n\n", "\n", ". ", " ", ""]`** (in priority order)

### Why this strategy?

The recursive splitter tries to split on the largest semantic unit first (paragraphs `\n\n`), and only falls back to smaller units (lines, sentences, words, characters) when a chunk is still too big. This **preserves semantic coherence** within each chunk far better than naive fixed-size slicing that cuts mid-sentence.

The **150-char overlap** ensures that information sitting on a chunk boundary isn't lost to retrieval — when a fact spans the cut, both neighboring chunks contain it.

Per-page chunking preserves `page_number` metadata, which is then surfaced as `[p. N]` citations in the final answer.

---

## 🚀 Quickstart (Local)

### 1. Get free API keys

| Service | Where | Credit card? |
|---|---|---|
| Gemini API key | https://aistudio.google.com/apikey | ❌ No |
| Qdrant Cloud cluster | https://cloud.qdrant.io/ → create free 1 GB cluster | ❌ No |

After creating the Qdrant cluster, copy its **REST URL** and **API key** from the cluster dashboard.

### 2. Clone & install

```bash
git clone <your-repo-url>
cd RAG-3
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure secrets

```bash
cp .env.example .env
# Then open .env and fill in your three keys.
```

### 4. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload a PDF, and ask away.

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this repo to **GitHub** (public).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"Create app"** → pick this repo → main file: `app.py`.
4. Open **"Advanced settings" → Secrets**, paste:

   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   QDRANT_URL = "https://your-cluster-id.region.cloud.qdrant.io:6333"
   QDRANT_API_KEY = "your_qdrant_api_key_here"
   ```

5. Click **Deploy**. Done — share the live URL.

The app will be at `https://<your-app-name>.streamlit.app`.

---

## 📁 Project structure

```
RAG-3/
├── app.py                      # Streamlit UI + chat orchestration
├── rag.py                      # RAG pipeline (chunk → embed → store → retrieve → generate)
├── requirements.txt            # Python deps
├── .env.example                # Local env template
├── .streamlit/
│   ├── config.toml             # Streamlit theme + upload size
│   └── secrets.toml.example    # Streamlit secrets template
├── .gitignore
└── README.md
```

---

## 🔬 How grounding works

The system prompt (`rag.SYSTEM_PROMPT`) is strict:

> Answer using ONLY the context excerpts below. If the answer is not in the context, say "I couldn't find that in the document." Do NOT use outside knowledge.

Combined with `temperature=0.2` and the fact that the model is *only* given the top-5 retrieved chunks (not the whole doc, not the web), this keeps answers tied to the document. The UI then shows every retrieved chunk in an expandable "Sources" panel so you can verify the grounding yourself.

---

## 🧪 Test it

Try uploading any PDF (research paper, contract, textbook chapter) and ask:
- *"Summarize the main argument."*
- *"What does the author say about X on page Y?"*
- *"What's the conclusion?"*

Then ask something the document doesn't cover — it should reply *"I couldn't find that in the document."* That's grounding working correctly.

---

## 📜 License

MIT — do whatever you want.
