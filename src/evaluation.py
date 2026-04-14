"""RAG pipeline evaluation utilities.

Provides keyword-recall evaluation for measuring how well the RAG pipeline
answers known questions.  Useful for tuning chunk size, overlap, top_k,
and prompt templates.
"""

import logging
from dataclasses import dataclass

from .rag_engine import RAGEngine

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    question: str
    answer: str
    expected_keywords: list[str]
    keyword_hits: int
    keyword_total: int
    recall: float
    has_sources: bool


def evaluate_rag(
    engine: RAGEngine,
    test_cases: list[dict],
) -> list[EvalResult]:
    """Run evaluation on a list of test cases.

    Each test case is a dict with:
        - ``question``: the query string
        - ``expected_keywords``: list of strings expected in a good answer
    """
    results: list[EvalResult] = []
    for case in test_cases:
        question = case["question"]
        expected = [kw.lower() for kw in case["expected_keywords"]]

        response = engine.query(question)
        answer_lower = response["answer"].lower()

        hits = sum(1 for kw in expected if kw in answer_lower)
        recall = hits / len(expected) if expected else 0.0

        results.append(
            EvalResult(
                question=question,
                answer=response["answer"],
                expected_keywords=case["expected_keywords"],
                keyword_hits=hits,
                keyword_total=len(expected),
                recall=round(recall, 2),
                has_sources=len(response["sources"]) > 0,
            )
        )

    avg_recall = sum(r.recall for r in results) / len(results) if results else 0
    logger.info(
        "Evaluation complete: %d cases, avg keyword recall=%.2f",
        len(results),
        avg_recall,
    )
    return results


def print_eval_report(results: list[EvalResult]) -> None:
    """Print a formatted evaluation report to stdout."""
    print("\n" + "=" * 70)
    print("RAG EVALUATION REPORT")
    print("=" * 70)

    for i, r in enumerate(results, 1):
        status = "PASS" if r.recall >= 0.5 else "FAIL"
        print(f"\n[{status}] Q{i}: {r.question}")
        print(f"  Answer:   {r.answer[:100]}{'...' if len(r.answer) > 100 else ''}")
        print(f"  Recall:   {r.keyword_hits}/{r.keyword_total} ({r.recall:.0%})")
        print(f"  Sources:  {'Yes' if r.has_sources else 'No'}")

    avg = sum(r.recall for r in results) / len(results) if results else 0
    passed = sum(1 for r in results if r.recall >= 0.5)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {passed}/{len(results)} passed | Avg recall: {avg:.0%}")
    print("=" * 70)
