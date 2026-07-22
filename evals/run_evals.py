"""Run the self-authored eval suite and print a pass rate.

    python evals/run_evals.py

Each case checks the capabilities in the rubric: grounding/refusal, filtering,
comparison, single-entity lookup and unit handling. Checks are intentionally
lenient on wording but strict on the things that matter (answered flag,
which college_ids are / aren't cited).
"""

import json
import os
import sys

# Allow running from repo root or from inside evals/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import answer_question  # noqa: E402

CASES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_cases.json")


def check_case(case, parsed):
    failures = []

    if "expect_answered" in case and parsed.answered != case["expect_answered"]:
        failures.append(f"answered={parsed.answered}, expected {case['expect_answered']}")

    cited = set(parsed.citations or [])

    for cid in case.get("must_cite", []):
        if cid not in cited:
            failures.append(f"missing citation {cid}")

    for cid in case.get("must_not_cite", []):
        if cid in cited:
            failures.append(f"should not cite {cid}")

    contains_any = case.get("answer_contains_any")
    if contains_any:
        text = parsed.answer.lower()
        if not any(sub.lower() in text for sub in contains_any):
            failures.append(f"answer missing any of {contains_any}")

    return failures


def main():
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    passed = 0
    for case in cases:
        parsed, _meta = answer_question(case["question"])
        failures = check_case(case, parsed)
        status = "PASS" if not failures else "FAIL"
        if not failures:
            passed += 1
        print(f"[{status}] {case['id']}")
        if failures:
            for fail in failures:
                print(f"        - {fail}")
            print(f"        got: answered={parsed.answered}, citations={parsed.citations}")

    total = len(cases)
    print(f"\nPass rate: {passed}/{total} = {round(100 * passed / total)}%")


if __name__ == "__main__":
    main()
