"""Core RAG logic for the Make My Education college counsellor.

This module owns the retrieval + grounded generation pipeline. Both the CLI
(`answer.py`) and the batch scripts (`run_all.py`, `evals/`) import from here so
there is a single source of truth.
"""

import os
import time
from typing import List, Optional

import chromadb
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# --- Configuration -----------------------------------------------------------
EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "gemini-3.5-flash-lite"
CHROMA_PATH = "./my_chroma_db"
COLLECTION_NAME = "college_collection"

# We retrieve generously: with only 15 tiny records, giving the model a wide
# candidate set removes the "the right college was never retrieved" failure and
# lets the model do the filtering/comparison reasoning.
N_RESULTS = 10


# --- Structured output schema (matches the assignment spec exactly) ----------
class Answer(BaseModel):
    answer: str = Field(description="The grounded answer, or 'no relevant answer found'.")
    citations: List[str] = Field(default_factory=list, description="college_id values the answer relies on.")
    answered: bool = Field(description="True if the data supports an answer, else False.")
    reason_if_unanswered: Optional[str] = Field(
        default=None, description="Short explanation when answered is False; otherwise null."
    )


SYSTEM_INSTRUCTION = """You are the Make My Education AI counsellor. You answer questions about colleges using ONLY the retrieved college records supplied in the user message.

GROUNDING (most important):
- Use ONLY facts present in the provided records. Never use outside knowledge and never invent a fact (fees, cutoffs, courses, placements).
- Do NOT infer or assert the absence of something. If the records do not EXPLICITLY contain the information the question asks for (e.g. a specific program, degree or field that is not listed), treat it as unanswerable: set answered=false, set answer to exactly "no relevant answer found", leave citations empty ([]), and in reason_if_unanswered briefly explain what is missing (you may note what the records do list).
- Only when the records explicitly support an answer: set answered=true, put the grounded answer in answer, set reason_if_unanswered to null, and cite the college_id (e.g. C002) of EVERY college the answer relies on. Each record is prefixed with its id in [square brackets].

DATA SEMANTICS (apply these carefully):
- last_year_cutoff_pct is a HARD FLOOR. A student scoring below it was NOT eligible. Never recommend a college whose cutoff is above the student's score.
- annual_fees_inr is PER YEAR, not per semester. If the user states a per-semester or total-course budget, convert it (1 year = 2 semesters) and state your assumption in the answer.
- avg_placement_lpa = 0 means "not reported / not applicable", NOT the worst placement. Never call it the worst.
- A Diploma is NOT a degree. When asked about degree/engineering colleges, treat diploma-only colleges as a judgment call and flag the distinction rather than silently including or excluding them.
- Some college names are deliberately similar. Match the exact college asked about; do not confuse two different colleges.
- annual_fees_inr excludes hostel, mess and other charges that may be described in the free-text about field. For budget questions, mention such extra charges when the records describe them.

STYLE: concise, polite, easy to understand. Include the relevant numbers (fees, cutoff, placement) and always cite college_id."""


# --- Lazy singletons (created once, reused across many questions) ------------
_client = None
_collection = None


def get_client():
    """Return a cached Gemini client, reading the key from the environment."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key found. Set GEMINI_API_KEY (PowerShell: $env:GEMINI_API_KEY=\"...\")."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def get_collection():
    """Return a cached handle to the persisted Chroma collection."""
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = chroma_client.get_collection(name=COLLECTION_NAME)
    return _collection


def _build_context(res) -> str:
    """Turn Chroma results into an id-tagged text block for the prompt."""
    ids = res["ids"][0]
    docs = res["documents"][0]
    blocks = [f"[{cid}]\n{doc}" for cid, doc in zip(ids, docs)]
    return "\n\n".join(blocks)


def answer_question(question: str, n_results: int = N_RESULTS):
    """Run the full pipeline for one question.

    Returns (answer: Answer, meta: dict) where meta carries token counts and
    latency for the Part D cost report.
    """
    client = get_client()
    collection = get_collection()

    t0 = time.perf_counter()

    # 1. Embed the query (query-side task type, distinct from the stored docs).
    query_em = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vector = query_em.embeddings[0].values

    # 2. Retrieve candidate records.
    res = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    context = _build_context(res)

    # 3. Generate a grounded, structured answer.
    contents = (
        f"User question: {question}\n\n"
        f"Retrieved college records (cite these college_ids only):\n{context}"
    )
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=Answer,
        temperature=0.0,
    )
    response = client.models.generate_content(
        model=GENERATION_MODEL, contents=contents, config=config
    )

    latency = time.perf_counter() - t0
    parsed = Answer.model_validate_json(response.text)

    # Token usage for the cost report (guarded: field names vary by SDK version).
    usage = getattr(response, "usage_metadata", None)
    meta = {
        "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
        "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
        "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
        "latency_s": round(latency, 3),
        "retrieved_ids": res["ids"][0],
        "top_distance": res["distances"][0][0] if res["distances"][0] else None,
    }
    return parsed, meta


# Set your key before running (PowerShell): $env:GEMINI_API_KEY="<your_key>"

# Question 1
# .\.venv\Scripts\python.exe answer.py "I scored 78% and have a budget of ₹1.5 lakh/year — which engineering colleges can I consider?"

# Question 5
# .\.venv\Scripts\python.exe answer.py "Does Ganga Valley University offer a PhD in Physics?"

# Evaluations
# .\.venv\Scripts\python.exe evals\run_evals.py