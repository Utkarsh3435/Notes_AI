"""
NotesAI — Handwritten PDF Q&A System
FastAPI backend: OCR → Embeddings → ChromaDB → Ollama RAG
"""

import os
import uuid
import json
import base64
from pathlib import Path
from typing import Optional
from llm import generate_important_questions
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from ocr import extract_text_from_pdf  # noqa: E402
from embeddings import get_embedding, get_embeddings_batch  # noqa: E402
from vectorstore import VectorStore  # noqa: E402
from llm import generate_answer, improve_query, generate_report_llm  # noqa: E402

app = FastAPI(title="NotesAI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Global vector store
vector_store = VectorStore()

# Conversation memory (last N turns)
conversation_history: list[dict] = []
MAX_HISTORY = 6

# Loaded PDFs registry
loaded_pdfs: list[dict] = []


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float
    has_answer: bool


@app.get("/")
def serve_frontend():
    return FileResponse(str(frontend_path / "index.html"))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF, run OCR on each page, chunk the text,
    embed each chunk, and store in ChromaDB.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_id = str(uuid.uuid4())[:8]
    pdf_bytes = await file.read()

    print(
        f"[UPLOAD] Processing '{file.filename}' ({len(pdf_bytes)} bytes) → id={pdf_id}"
    )

    try:
        # Step 1: OCR — extract text page by page
        pages = extract_text_from_pdf(pdf_bytes, pdf_id=pdf_id, filename=file.filename)
        total_pages = len(pages)
        print(f"[OCR] Extracted {total_pages} pages")

        # Step 2: Chunk all pages first
        chunks_added = 0
        all_chunks = []

        for page in pages:
            chunks = chunk_text(page["text"], page_num=page["page_num"])

            for chunk in chunks:
                if len(chunk["text"].strip()) < 20:
                    continue

                chunk["filename"] = file.filename
                chunk["pdf_id"] = pdf_id
                chunk["total_pages"] = total_pages

                all_chunks.append(chunk)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No usable text found in PDF.")

        # Step 3: Batch embed all chunks
        texts = [c["text"] for c in all_chunks]
        embeddings = get_embeddings_batch(texts)

        # Step 4: Store in vector DB
        for chunk, embedding in zip(all_chunks, embeddings):
            vector_store.add(
                doc_id=f"{chunk['pdf_id']}_p{chunk['page_num']}_c{chunk['chunk_idx']}",
                text=chunk["text"],
                embedding=embedding,
                metadata={
                    "pdf_id": chunk["pdf_id"],
                    "filename": chunk["filename"],
                    "page_num": chunk["page_num"],
                    "chunk_idx": chunk["chunk_idx"],
                    "total_pages": chunk["total_pages"],
                },
            )

            chunks_added += 1

        loaded_pdfs.append(
            {
                "pdf_id": pdf_id,
                "filename": file.filename,
                "pages": total_pages,
                "chunks": chunks_added,
            }
        )

        print(f"[STORE] Added {chunks_added} chunks from '{file.filename}'")

        return {
            "status": "success",
            "pdf_id": pdf_id,
            "filename": file.filename,
            "pages": total_pages,
            "chunks_indexed": chunks_added,
        }

    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_questions")
def generate_questions():

    results = vector_store.collection.get()

    texts = results["documents"][:20]  # limit to avoid huge prompt

    questions = generate_important_questions(texts)

    return {"questions": questions}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    RAG pipeline:
      1. Embed the user's question
      2. Retrieve top_k relevant chunks from ChromaDB
      3. Build a prompt with context + conversation history
      4. Run Ollama local LLM
      5. Return answer + source references + confidence score
    """
    global conversation_history

    if vector_store.count() == 0:
        return ChatResponse(
            answer="No notes have been uploaded yet. Please upload a PDF first.",
            sources=[],
            confidence=0.0,
            has_answer=False,
        )

    # Step 1: Embed the improved question
    improved_q = improve_query(req.question)
    q_embedding = get_embedding(improved_q)

    # Step 2: Retrieve relevant chunks
    results = vector_store.search(q_embedding, top_k=req.top_k)

    if not results:
        return ChatResponse(
            answer="I don't have enough information in the notes to answer that.",
            sources=[],
            confidence=0.0,
            has_answer=False,
        )

    # Step 3: Build context block
    context_blocks = []
    for i, r in enumerate(results):
        context_blocks.append(
            f"[Source {i + 1} — {r['metadata']['filename']}, Page {r['metadata']['page_num']}]\n{r['text']}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    # Step 4: Build conversation history string
    history_str = ""

    if conversation_history:
        history_str = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in conversation_history[-MAX_HISTORY:]
        )

    # Step 5: Generate answer with local LLM
    scores = [r["score"] for r in results]

    answer, confidence = generate_answer(
        question=req.question,
        context=context,
        history=history_str,
        retrieval_scores=scores,
    )

    # Update conversation memory
    conversation_history.append({"role": "user", "content": req.question})
    conversation_history.append({"role": "assistant", "content": answer})

    # Build source references (deduplicated by page)
    seen = set()
    sources = []

    for r in results:
        key = (r["metadata"]["filename"], r["metadata"]["page_num"])

        if key not in seen:
            seen.add(key)

            sources.append(
                {
                    "filename": r["metadata"]["filename"],
                    "page": r["metadata"]["page_num"],
                    "excerpt": r["text"][:200] + ("…" if len(r["text"]) > 200 else ""),
                    "score": round(r["score"], 3),
                }
            )

    has_answer = (
        "don't have" not in answer.lower() and "not in the notes" not in answer.lower()
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        has_answer=has_answer,
    )


@app.post("/generate_report")
def generate_report(data: dict):

    pdf_id = data["pdf_id"]

    results = vector_store.collection.get(where={"pdf_id": pdf_id})

    texts = results["documents"][:20]

    combined_text = "\n".join(texts)

    report = generate_report_llm(combined_text)

    return {"report": report}


@app.get("/pdfs")
def list_pdfs():
    return {"pdfs": loaded_pdfs}


@app.delete("/reset")
def reset():
    """Clear all indexed data and conversation history."""
    vector_store.clear()
    conversation_history.clear()
    loaded_pdfs.clear()
    return {"status": "reset complete"}


# ── Helpers ──────────────────────────────────────────────────────────────────


def chunk_text(
    text: str, page_num: int, chunk_size: int = 120, overlap: int = 30
) -> list[dict]:
    """Split text into overlapping chunks."""

    words = text.split()
    chunks = []
    i = 0
    idx = 0

    while i < len(words):
        chunk_words = words[i : i + chunk_size]

        if len(chunk_words) < 20:
            break

        chunks.append(
            {
                "text": " ".join(chunk_words),
                "chunk_idx": idx,
                "page_num": page_num,
            }
        )

        i += chunk_size - overlap
        idx += 1

    return chunks if chunks else [{"text": text, "chunk_idx": 0, "page_num": page_num}]


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
