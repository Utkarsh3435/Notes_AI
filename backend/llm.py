import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3"


def improve_query(question: str) -> str:
    """Expand the user's question to improve vector search."""

    prompt = f"""
Rewrite the student's question as a short keyword search query for study notes retrieval.

Focus on technical keywords.

Question:
{question}

Search query:
"""

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except:
        return question


def generate_report_llm(context):

    prompt = f"""
You are an AI study assistant.

Create a structured report of the following notes.

Include:

1. Main topics
2. Key concepts
3. Important definitions
4. Short summary

Notes:
{context}

Report:
"""

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["message"]["content"]


def generate_important_questions(context):

    context_text = "\n\n".join(context)

    prompt = f"""
You are an AI exam preparation assistant.

Based ONLY on the following study notes, generate important exam questions.

Rules:
- Do not use outside knowledge
- Focus on concepts from the notes
- Generate:
  • 5 short answer questions
  • 5 long answer questions
- Questions should be useful for university exams.

Notes:
{context_text}

Important Questions:
"""

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["message"]["content"]


def generate_answer(question, context, history="", retrieval_scores=None):

    context_text = "\n\n".join(
        [
            f"Note Section {i + 1} (from student notes):\n{c}"
            for i, c in enumerate(context)
        ]
    )

    prompt = f"""
You are answering questions from a student's handwritten notes.

You MUST extract information ONLY from the provided context.

STRICT RULES:
- Do NOT use outside knowledge.
- Do NOT add components not present in the context.
- If the answer is not clearly present in the context, say:
  "Answer not found in the notes."
- Copy the concepts directly from the notes and organize them clearly.

Return the answer in bullet points suitable for exam preparation.

CONTEXT FROM STUDENT NOTES:
{context_text}

QUESTION:
{question}

FINAL ANSWER (only using context):
"""

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    data = response.json()
    answer = data["message"]["content"]

    if retrieval_scores:
        confidence = round(sum(retrieval_scores) / len(retrieval_scores), 2)
    else:
        confidence = 0.5

    return answer, confidence
