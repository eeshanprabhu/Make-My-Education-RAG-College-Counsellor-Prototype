"""Regenerate answers.md for the seven published questions in one command.

    python run_all.py

Also prints a Part D cost/latency summary (averaged over the seven questions)
to stdout so the numbers in the README are measured, not guessed.
"""

import json
import statistics

from rag import GENERATION_MODEL, answer_question

PUBLISHED_QUESTIONS = [
    "I scored 78% and have a budget of \u20b91.5 lakh/year \u2014 which engineering colleges can I consider?",
    "Which colleges offer an MBA, and what do they cost?",
    "List the government colleges that have hostel facilities.",
    "What\u2019s the average placement package at North Ridge Institute of Technology?",
    "Does Ganga Valley University offer a PhD in Physics?",
    "Which colleges offer scholarships for students from low-income families?",
    "Which college is best for me? I have \u20b91 lakh per semester.",
]


def main():
    blocks = ["# Answers \u2014 seven published questions\n",
              "> Verbatim output of `answer.py` for each published question. Not edited.\n"]
    input_tokens, output_tokens, latencies = [], [], []

    for q in PUBLISHED_QUESTIONS:
        parsed, meta = answer_question(q)
        payload = json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2)
        blocks.append(f"### Q: {q}\n\n```json\n{payload}\n```\n")

        if meta["input_tokens"]:
            input_tokens.append(meta["input_tokens"])
        if meta["output_tokens"]:
            output_tokens.append(meta["output_tokens"])
        latencies.append(meta["latency_s"])

    with open("answers.md", "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))

    # ---- Part D: measured cost summary ----
    def avg(xs):
        return round(statistics.mean(xs), 1) if xs else float("nan")

    print("answers.md regenerated.\n")
    print("=== Measured cost/latency (avg over 7 questions) ===")
    print(f"Model:                 {GENERATION_MODEL}")
    print(f"Avg input tokens:      {avg(input_tokens)}")
    print(f"Avg output tokens:     {avg(output_tokens)}")
    print(f"Avg latency (s):       {avg(latencies)}")
    print("(Plug your provider's per-1M-token price into README Part D to get \u20b9/1k queries.)")


if __name__ == "__main__":
    main()
