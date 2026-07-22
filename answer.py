"""CLI entry point required by the assignment.

Usage:
    python answer.py "Which colleges offer an MBA, and what do they cost?"

Prints exactly ONE JSON object to stdout with keys:
    answer, citations, answered, reason_if_unanswered
"""

import json
import sys

from rag import answer_question


def main():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        # Keep stdout as a single JSON object even on misuse.
        print(json.dumps({
            "answer": "no relevant answer found",
            "citations": [],
            "answered": False,
            "reason_if_unanswered": "No question was provided. Usage: python answer.py \"<question>\"",
        }, ensure_ascii=False))
        sys.exit(1)

    question = sys.argv[1]
    parsed, _meta = answer_question(question)

    # stdout must be ONLY the JSON object (graders parse this).
    print(json.dumps(parsed.model_dump(), ensure_ascii=False))


if __name__ == "__main__":
    main()
