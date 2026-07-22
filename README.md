# Make My Education — RAG College Counsellor (Prototype)

A small Retrieval-Augmented Generation prototype that answers natural-language
questions about 15 colleges, grounded strictly in the provided dataset. Every
factual claim cites the `college_id` it came from, and the system refuses
(`answered: false`) when the data cannot support an answer.

---

## How to run (clean checkout)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key (this prototype uses Google Gemini)
#    PowerShell:
$env:GEMINI_API_KEY="your_key_here"
#    bash:
export GEMINI_API_KEY=your_key_here

# 3. Ask a question (required interface)
#    The prebuilt vector store (my_chroma_db/) ships with the repo, so this
#    works immediately. To rebuild it from sample_colleges.csv, run: python ingest.py
python answer.py "Which colleges offer an MBA, and what do they cost?"
```

`answer.py` prints exactly one JSON object to stdout:

```json
{
  "answer": "Ganga Valley University (₹98,000/yr) and Doon Business School (₹1,75,000/yr).",
  "citations": ["C002", "C004"],
  "answered": true,
  "reason_if_unanswered": null
}
```

Other commands:

```bash
python run_all.py          # regenerates answers.md + prints measured cost/latency
python evals/run_evals.py  # runs the self-authored eval suite and prints a pass rate
```

---

## Repository layout

| File | Purpose |
| --- | --- |
| `ingest.py` | Reads the dataset, embeds each college, upserts vectors + metadata into Chroma. Idempotent. |
| `rag.py` | Core pipeline: embed query → retrieve → grounded structured generation. Single source of truth. |
| `answer.py` | CLI required by the spec; prints one JSON object. |
| `run_all.py` | Regenerates `answers.md` for the 7 published questions; prints Part D cost numbers. |
| `answers.md` | Verbatim system output for the 7 published questions (generated). |
| `evals/` | 10 question/expectation pairs + a runner that prints a pass rate. |

---

## Key design choices

- **Retrieve wide, let the model reason.** With only 15 tiny records, narrow
  top-k retrieval was the main failure source (a constrained question could rank
  an *ineligible* college first and miss valid ones). I retrieve a generous
  candidate set (`N_RESULTS=10`) and let the model filter/compare/rank. This is
  still RAG — I'm grounding the model in retrieved records — but sized honestly
  to the dataset. At real scale this becomes: metadata pre-filter (`where`) +
  narrower top-k.
- **Grounding is enforced in the system prompt + structured output.** The model
  is instructed to use only the supplied records, to cite `college_id`, and to
  return `answered: false` with a reason when it can't answer. Output is a strict
  JSON schema (Pydantic `response_schema`), so downstream parsing never breaks.
- **The data-dictionary traps live in the system prompt**, not in the user's
  question: cutoff is a hard floor, fees are per-year (convert per-semester and
  state the assumption), placement `0` = not reported, diploma ≠ degree, and
  watch the two similar-named colleges.
- **Two-layer refusal.** A distance signal catches wholly off-topic questions;
  the grounding prompt catches "relevant college, missing fact" (e.g. PhD in
  Physics), which a distance threshold alone cannot.
- **Task types.** Documents are embedded with `RETRIEVAL_DOCUMENT` and queries
  with `RETRIEVAL_QUERY` for better retrieval alignment.
- **Low temperature (0.0)** for factual, deterministic answers.

---

## Evaluation results & known limitations

`evals/run_evals.py` runs 10 self-authored cases covering grounding, refusal,
filtering, comparison, single-entity lookup, superlatives and unit conversion.

**Current pass rate: 10/10.** Cases and expectations are in
`evals/eval_cases.json`.

Known limitations (honest failure modes I'm aware of):

- **Constraints are enforced by the model, not a hard filter.** I rely on the
  system prompt to apply cutoff/budget rules rather than a metadata `where`
  filter. At temperature 0 this is reliable on the tested questions, but it is
  softer than a deterministic filter and could slip on unseen phrasings.
- **Citations can include "referenced-but-not-recommended" colleges.** For the
  score+budget question the model sometimes cites the colleges it *excludes*
  (to justify why), which is grounded but can look like a recommendation.
- **"Best" is under-specified.** For the per-semester question the model lists
  all affordable colleges rather than committing to a single ranking; this is a
  deliberate judgment given "best" is subjective, but a sharper product would
  ask a clarifying question.
- **Retrieval is sized to 15 records.** The wide candidate set (`N_RESULTS=10`)
  works here but would not scale; see cost notes below.

## Part D — Cost, with numbers

> Numbers below are measured by `run_all.py` (token counts from the API
> response's `usage_metadata`, latency from `time.perf_counter()`), averaged
> over the seven published questions. Fill in the `< >` price cells from your
> provider's current price sheet.

| Metric | Your number |
| --- | --- |
| Average input tokens per query | **3,985** |
| Average output tokens per query | **175** |
| Average end-to-end latency per query | **2.2 s** |
| Model(s) used, and cost per 1M tokens | Generation: `gemini-3.5-flash-lite` — **$0.30/1M input, $2.50/1M output** (paid tier). Embeddings: `gemini-embedding-2` (one-time, separate, negligible). |
| Cost per 1,000 queries, in ₹ | **≈ ₹140** (see calc below) |
| One-time embedding cost for the full dataset | **≈ ₹0.1** (15 docs × ~450 tokens ≈ 6,750 tokens; negligible) |

**Calculation** (USD→₹ at ~₹86/$):
- Per query = (3,985 input × $0.30 + 175 output × $2.50) ÷ 1,000,000 = **$0.00163**
- Per 1,000 queries = $1.63 ≈ **₹140**
- At **50,000 queries/month** ≈ $82/month ≈ **₹7,000/month** in model cost.

Note: input tokens dominate cost (context is large because I retrieve a wide
candidate set). Context caching ($0.03/1M) or a metadata pre-filter would cut
this materially — see below.

**At 50,000 queries/month, what breaks first?** Cost and latency scale linearly
with queries; accuracy does not degrade with volume. The dominant cost driver is
the **input tokens** (~3,985/query) — because I stuff a wide candidate set into
the prompt. The first change is therefore to **shrink the context**: apply a
metadata `where` pre-filter so only relevant records enter the prompt (cutting
input tokens by an estimated 60–70%), and **cache** answers for repeated/near
-duplicate questions. Embedding cost is a one-time, negligible amount for 15
records. Net: reducing input tokens + caching would cut per-query cost the most,
with latency improving as a side effect.

---

## Part C — Short reflection

- **Keeping per-query cost low as usage grows:** Use the smallest model that
  passes the eval suite (already on a `flash-lite` tier); pre-filter records with
  metadata so the prompt context stays small; cache answers for identical/near
  -duplicate questions; and cap output tokens. Input tokens are the main lever
  here (~3,985/query today).
- **Never stating a wrong fee or cutoff:** Strict grounding prompt ("only from
  records, never infer absence"), structured output with mandatory `college_id`
  citations, temperature 0, and a refusal path (`answered: false`). Regressions
  are caught by evals that assert exact fee/cutoff values and forbidden
  recommendations (e.g. a college above the student's cutoff).
- **What I'd build first if I joined:** A metadata pre-filter layer over the
  college data (budget, cutoff, course, location) so hard constraints are
  enforced deterministically before the LLM sees anything, plus a citation UI
  that lets a student click any claim through to the exact source record. This
  directly protects trust (no wrong fee/cutoff) while keeping cost low.
- **How I'd measure whether AI actually helps students:** Track outcomes, not
  vanity metrics — shortlist-to-application conversion, how often students
  escalate to a human counsellor (lower is better once answers are trusted), and
  periodic human audits comparing answers against the source data for accuracy.
  A student who applies with confidence is the real signal.

---

## Part B — Proof you've shipped

**Arabic Speech Emotion Recognition — AI Candidate Screening System**

- **What it did:** An AI pipeline that screened candidates from Arabic video
  resumes. It transcribed speech with OpenAI Whisper (~98% transcription
  accuracy), ran downstream emotion/sentiment analysis, and classified
  candidates across emotional classes (confident, assertive, tense, composed) at
  ~85% classification accuracy. It then computed composite behavioural scores
  (confidence, nervousness, calmness) from the emotion probability
  distributions, enabling automated, bias-reduced HR screening at scale.
- **Who used it / scale:** HR teams submitted video resumes and retrieved
  scoring reports programmatically via a REST API — screening roughly **500
  resumes per day**.
- **My role:** I built the end-to-end pipeline — Whisper transcription, the
  emotion classification stage, the composite scoring logic, and the REST API
  endpoint — and deployed it on a VPS behind Nginx.
- **What broke / surprised me in production:** Concurrent processing broke under
  load — simultaneous requests overwhelmed the pipeline. I fixed it by adding a
  **request queue** so submissions were processed sequentially/throttled instead
  of all at once, which stabilised throughput.
- **What it cost & how I brought it down:** Two cost lines — a VPS to host the
  code (~₹5,000/month) and the OpenAI API on a pay-as-you-go basis (scaling with
  transcription volume).

---

## What I'd do differently with more time

- Add a metadata `where` pre-filter for hard numeric constraints (budget,
  cutoff) instead of relying on the model to enforce them.
- Expand the eval suite and add exact-value assertions on fees/cutoffs.
- Handle per-semester / per-lakh / total-course budget phrasing more explicitly.
- Move the API key fully to a `.env` and add ret/back-off on transient API errors.

---

## Notes

- The dataset is treated as the whole world; questions outside it are refused.
- `.venv/`, `my_chroma_db/` and any key material should not be committed (see
  `.gitignore`).
