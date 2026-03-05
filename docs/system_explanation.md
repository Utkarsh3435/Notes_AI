# NotesAI — System Explanation
### Gen AI Hackathon 2026 · Document Intelligence Track

---

## What the System Does

NotesAI is a **Retrieval-Augmented Generation (RAG)** pipeline built specifically for handwritten PDF notes. Given one or more PDFs, the system:

1. Extracts handwritten text from every page using AI-powered OCR
2. Stores the text in a searchable local vector database
3. Accepts natural-language questions from the user
4. Retrieves the most relevant sections of the notes
5. Generates a concise, cited answer using a locally running language model

The system **never invents information**. Every answer is grounded in the retrieved text, and the model is explicitly instructed to say "I don't have enough information" when the notes don't cover a topic.

---

## Step-by-Step Pipeline

### Step 1 — OCR (Cloud, allowed)
Each page of the uploaded PDF is rendered as a JPEG image at 200 DPI using `pdf2image`. The image is sent to **Anthropic Claude Vision** with a strict transcription prompt. Claude Vision was chosen because it handles:
- Messy, inconsistent handwriting
- Tables and labelled diagrams
- Tilted or poorly scanned pages

The prompt instructs the model to output only verbatim transcription — no summaries or additions.

### Step 2 — Chunking
Each page's transcribed text is split into overlapping chunks of ~400 words (with 80-word overlap). Overlap prevents context loss at chunk boundaries and helps retrieve complete answers even when an answer spans a chunk edge.

### Step 3 — Embedding (Local)
Each chunk is converted to a 384-dimensional vector using **`all-MiniLM-L6-v2`** from the `sentence-transformers` library. This model runs entirely locally, requires no internet after the initial ~80 MB download, and produces high-quality semantic embeddings suitable for question-answering retrieval tasks.

### Step 4 — Vector Storage (Local)
All embeddings and their associated metadata (filename, page number, chunk index) are stored in **ChromaDB**, a persistent local vector database. ChromaDB uses HNSW indexing with cosine distance, enabling fast approximate nearest-neighbour search over thousands of chunks.

### Step 5 — Question Embedding & Retrieval
When the user asks a question, it is embedded using the same `all-MiniLM-L6-v2` model. The resulting vector is queried against ChromaDB to retrieve the top-5 most semantically similar chunks. Chunks with cosine similarity below 0.25 are filtered out to avoid returning irrelevant context.

### Step 6 — Answer Generation (Local LLM)
The retrieved chunks, together with the last 6 turns of conversation history, are assembled into a structured prompt and sent to **Ollama** running `llama3.2` on the local machine. The system prompt strictly forbids the model from using outside knowledge — it must answer from the provided context only. Temperature is set to 0.1 to minimise hallucination.

### Step 7 — Response + Citations
The answer is returned to the frontend alongside:
- **Source tags**: each unique (filename, page) pair that contributed to the answer
- **Confidence score**: a heuristic based on how many key answer terms appear in the retrieved context
- A flag indicating whether the model declined to answer (used to style the response differently)

---

## Bonus Features Implemented

| Feature | Implementation |
|---|---|
| **Confidence scoring** | Term-overlap heuristic, displayed as a colour-coded bar |
| **Messy handwriting** | Claude Vision OCR far outperforms Tesseract on degraded handwriting |
| **Conversation memory** | Last 6 user/assistant turns included in every LLM prompt |
| **Clean, modular code** | Separate modules: `ocr.py`, `embeddings.py`, `vectorstore.py`, `llm.py`, `app.py` |
| **Live PDF upload** | New PDFs can be dropped during a demo and are immediately searchable |

---

## Technology Choices

| Component | Technology | Reason |
|---|---|---|
| OCR | Anthropic Claude Vision | Best-in-class handwriting recognition; cloud use explicitly allowed |
| Embeddings | sentence-transformers | Fast, local, no GPU required, high quality |
| Vector DB | ChromaDB | Zero-config, persistent, Python-native |
| Local LLM | Ollama + llama3.2 | Simple setup, runs on CPU, good instruction-following |
| Backend | FastAPI | Lightweight, async, auto-generates API docs |
| Frontend | Vanilla HTML/CSS/JS | Zero build step, single file, works offline |

---

## Why This Approach Wins

**Accuracy first**: the LLM is the last step, not the first. Retrieval quality determines answer quality — the model only amplifies what the notes actually say.

**Honest by design**: explicit "I don't know" handling, low temperature, strict system prompt, and confidence scoring all work together to eliminate hallucination.

**Demo-ready**: live PDF upload, persistent ChromaDB (survives restarts), and a polished UI mean the demo works smoothly under pressure.
