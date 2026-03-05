<<<<<<< HEAD
# NotesAI — Handwritten PDF Study Assistant

> Gen AI Hackathon 2026 · Document Intelligence Track

A fully local RAG system that reads handwritten PDF notes, indexes them, and answers questions — with cited sources and confidence scoring. No internet required after PDF upload.

---

## Architecture at a Glance

```
PDF Upload
    │
    ▼
[OCR — Claude Vision API]   ← Cloud step (allowed by rules)
    │  Extracts text from each page image
    ▼
[Chunking]                  ← Local
    │  Splits text into 400-word overlapping chunks
    ▼
[Embeddings — sentence-transformers]  ← Local (all-MiniLM-L6-v2)
    │  Converts each chunk to a 384-dim vector
    ▼
[ChromaDB]                  ← Local persistent vector store
    │  Stores chunks + embeddings on disk
    ▼
User Question
    │
    ▼
[Embed Question]  →  [ChromaDB cosine search]  →  Top-K chunks
    │
    ▼
[Ollama — llama3.2]         ← Local LLM
    │  Generates answer from context only
    ▼
Answer + Sources + Confidence Score
```

---

## Quick Start

### Prerequisites

| Tool | Install |
|---|---|
| Python ≥ 3.10 | https://python.org |
| Poppler (PDF→image) | `brew install poppler` / `apt install poppler-utils` |
| Ollama | https://ollama.com |

### 1 — Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2 — Pull the local LLM

```bash
ollama pull llama3.2
```

You can use any Ollama-compatible model. Edit `MODEL_NAME` in `backend/llm.py` to switch.

### 3 — Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

This is only used for the OCR step (explicitly permitted by hackathon rules).

### 4 — Start Ollama

```bash
ollama serve
```

### 5 — Start the backend

```bash
cd backend
python app.py
```

### 6 — Open the UI

Navigate to **http://localhost:8000** in your browser.

---

## Usage

1. **Upload** — drag a PDF onto the sidebar (or click to browse).  
   The system OCRs each page, chunks the text, embeds it, and stores it in ChromaDB.

2. **Ask** — type any question in the chat box and press Enter.  
   The system embeds your question, retrieves the most relevant chunks, and sends them to the local LLM.

3. **Read** — every answer shows:
   - The answer text
   - A confidence score (heuristic, based on term overlap)
   - Source tags: filename + page number + excerpt

4. **Repeat** — upload a second PDF live during the demo. New chunks are immediately searchable.

---

## Project Structure

```
notesai/
├── backend/
│   ├── app.py           Main FastAPI app (routing, chunking)
│   ├── ocr.py           Claude Vision OCR per page
│   ├── embeddings.py    sentence-transformers (local)
│   ├── vectorstore.py   ChromaDB wrapper (local)
│   ├── llm.py           Ollama local LLM + confidence scoring
│   ├── requirements.txt
│   └── chroma_db/       Created at runtime — persistent vector store
├── frontend/
│   └── index.html       Single-file dark-themed UI
├── docs/
│   └── system_explanation.md
└── README.md
```

---

## Scoring Alignment

| Criterion | How we address it |
|---|---|
| **Correct Answers (40%)** | Faithful RAG — LLM is strictly instructed to use only retrieved context. Low temperature (0.1) reduces hallucination. |
| **RAG Workflow (20%)** | Full pipeline: OCR → Chunking → Embeddings → Vector search → LLM generation |
| **Bonus Features (20%)** | ✓ Confidence scoring · ✓ Conversation memory · ✓ Messy handwriting handled by Claude Vision · ✓ Clean modular code |
| **UI Design (10%)** | Dark, typographically refined single-page app |
| **Presentation (10%)** | Clear step-by-step README, architecture diagram above |

---

## Configuration

| File | Variable | Default | Notes |
|---|---|---|---|
| `llm.py` | `MODEL_NAME` | `llama3.2` | Any Ollama model |
| `llm.py` | `OLLAMA_URL` | `http://localhost:11434/...` | Change port if needed |
| `embeddings.py` | `MODEL_NAME` | `all-MiniLM-L6-v2` | 384-dim, ~80 MB |
| `app.py` | `chunk_size` | `400` words | Smaller = finer retrieval |
| `app.py` | `overlap` | `80` words | Larger = less context loss at boundaries |
| `app.py` | `MAX_HISTORY` | `6` turns | Conversation memory window |
| `vectorstore.py` | `DB_PATH` | `./chroma_db` | Where vectors are persisted |

---

## FAQ

**Q: Why use Claude Vision for OCR?**  
The hackathon explicitly permits cloud APIs for the OCR step. Claude Vision handles messy handwriting, skewed pages, tables, and labelled diagrams far better than local OCR tools.

**Q: Does it work offline after upload?**  
Yes. Once a PDF is indexed, all subsequent operations (embedding queries, vector search, LLM generation) run entirely on your machine.

**Q: What if Ollama is slow?**  
Use a smaller model: `ollama pull phi3:mini` and update `MODEL_NAME = "phi3:mini"` in `llm.py`.

**Q: Can I add more PDFs during the demo?**  
Yes — just drag and drop. New chunks are appended to the existing ChromaDB collection.
=======
# Notes_AI
>>>>>>> 2ca2db161e213bbfeb6b1be0ee36d4bdf037198a
