"""CLI to run the four-dimension RAG evaluation matrix and gate regressions.

Usage::

    # Run eval and print the matrix
    python -m green_gov_rag.scripts.evaluate_rag --dataset eval/eval_dataset.json

    # Save the current run as the baseline
    python -m green_gov_rag.scripts.evaluate_rag --dataset eval/eval_dataset.json \
        --save-baseline eval/baseline.json

    # Run eval and fail (exit 1) if it regresses against the baseline
    python -m green_gov_rag.scripts.evaluate_rag --dataset eval/eval_dataset.json \
        --baseline eval/baseline.json

Retrieval hit rate + latency are always computed. Answer accuracy is computed when
``--with-answers`` is set (runs the full QueryService, slower / uses LLM). Satisfaction
is read from stored feedback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time

logger = logging.getLogger(__name__)


def _run_retrieval_eval(items, k: int):
    """Run retrieval for each item; return (hits, ranks, latencies_ms)."""
    from green_gov_rag.eval.dataset import _doc_file_id, _doc_text, item_is_hit
    from green_gov_rag.rag.agent_tools import RAGAgent

    agent = RAGAgent()
    hits: list[bool] = []
    ranks: list = []
    latencies: list[float] = []

    for item in items:
        start = time.perf_counter()
        _context, docs = agent.retrieve(
            query=item.query, metadata_filters=item.filters or None, k=k
        )
        latencies.append((time.perf_counter() - start) * 1000)

        hit = item_is_hit(item, docs)
        hits.append(hit)

        # First relevant rank for MRR.
        rank = None
        for i, doc in enumerate(docs, start=1):
            single = item_is_hit(item, [doc])
            if single:
                rank = i
                break
        ranks.append(rank)

    return hits, ranks, latencies


async def _run_answer_eval(items, judge_mode: str = "keyword"):
    """Run full QueryService per item and score answers.

    ``judge_mode``:
      - "keyword": cheap keyword-overlap proxy (original behaviour)
      - "llm":     LLM-as-judge correctness (+ faithfulness / citation as extras)
      - "both":    compute both, so keyword vs LLM-judge can be compared

    Returns a dict: ``{"keyword": [...], "llm": [...], "rows": [...]}`` where each
    row has per-item scores for side-by-side inspection.
    """
    from green_gov_rag.api.services.query_service import QueryService
    from green_gov_rag.eval.metrics import keyword_answer_score

    want_llm = judge_mode in ("llm", "both")
    want_kw = judge_mode in ("keyword", "both")
    if want_llm:
        from green_gov_rag.eval.llm_judge import judge_answer

    service = QueryService()
    keyword_scores: list[float] = []
    llm_scores: list[float] = []
    rows: list[dict] = []

    for item in items:
        # Keyword mode needs expected_keywords; LLM correctness needs a reference.
        if want_kw and not want_llm and not item.expected_keywords:
            continue

        resp = await service.execute_query(
            query=item.query,
            region=item.filters.get("region"),
            jurisdiction=item.filters.get("jurisdiction"),
            max_sources=5,
        )

        row: dict = {"id": item.id}

        if want_kw and item.expected_keywords:
            kw = keyword_answer_score(resp.answer, item.expected_keywords)
            keyword_scores.append(kw)
            row["keyword"] = round(kw, 3)

        if want_llm and item.reference_answer:
            sources = _sources_as_dicts(resp)
            contexts = [s["excerpt"] for s in sources if s.get("excerpt")]
            j = await judge_answer(
                query=item.query,
                answer=resp.answer,
                reference_answer=item.reference_answer,
                contexts=contexts,
                sources=sources,
            )
            llm_scores.append(j.correctness.score)
            row["llm_correctness"] = round(j.correctness.score, 3)
            row["llm_faithfulness"] = round(j.faithfulness.score, 3)
            row["llm_citation"] = round(j.citation_accuracy.score, 3)

        rows.append(row)

    return {"keyword": keyword_scores, "llm": llm_scores, "rows": rows}


def _sources_as_dicts(resp) -> list[dict]:
    """Normalise QueryResponse.sources into plain dicts for the judges."""
    out: list[dict] = []
    for s in getattr(resp, "sources", None) or []:
        if isinstance(s, dict):
            out.append(s)
        else:
            out.append(
                {
                    "title": getattr(s, "title", "") or "",
                    "excerpt": getattr(s, "excerpt", "") or getattr(s, "content", "") or "",
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation matrix")
    parser.add_argument("--dataset", required=True, help="Path to eval dataset JSON")
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval hit rate")
    parser.add_argument(
        "--with-answers",
        action="store_true",
        help="Also compute answer accuracy (runs full pipeline / LLM)",
    )
    parser.add_argument(
        "--judge",
        choices=["keyword", "llm", "both"],
        default="keyword",
        help="Answer scorer: keyword proxy, LLM-as-judge, or both (comparison). "
        "Only used with --with-answers.",
    )
    parser.add_argument("--baseline", help="Baseline JSON to regression-check against")
    parser.add_argument("--save-baseline", help="Save this run's matrix as baseline")
    parser.add_argument("--output", help="Write the matrix JSON to this path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from green_gov_rag.eval.dataset import load_dataset
    from green_gov_rag.eval.metrics import (
        EvalMatrix,
        compute_answer_accuracy,
        compute_latency_metrics,
        compute_retrieval_metrics,
        compute_satisfaction,
    )

    items = load_dataset(args.dataset)
    logger.info("Loaded %d eval items", len(items))

    hits, ranks, latencies = _run_retrieval_eval(items, args.k)
    retrieval = compute_retrieval_metrics(hits, ranks)
    latency = compute_latency_metrics(latencies)

    answer_accuracy = None
    if args.with_answers:
        answer_eval = asyncio.run(_run_answer_eval(items, judge_mode=args.judge))
        kw_scores = answer_eval["keyword"]
        llm_scores = answer_eval["llm"]

        # The matrix's answer_accuracy uses the LLM judge when available, else keyword.
        if llm_scores:
            answer_accuracy = compute_answer_accuracy(llm_scores)
        elif kw_scores:
            answer_accuracy = compute_answer_accuracy(kw_scores)

        # Side-by-side comparison when both scorers ran.
        if args.judge == "both":
            kw_mean = compute_answer_accuracy(kw_scores)
            llm_mean = compute_answer_accuracy(llm_scores)
            print("\n=== Answer Scoring: keyword vs LLM-judge ===")
            print(f"keyword mean:     {kw_mean if kw_mean is not None else 'n/a'}")
            print(f"llm mean:         {llm_mean if llm_mean is not None else 'n/a'}")
            print("per-item rows:")
            print(json.dumps(answer_eval["rows"], indent=2))

    satisfaction, sample = compute_satisfaction()

    matrix = EvalMatrix(
        retrieval_hit_rate=retrieval["hit_rate"],
        retrieval_mrr=retrieval["mrr"],
        answer_accuracy=answer_accuracy,
        latency_p50_ms=latency["p50"],
        latency_p95_ms=latency["p95"],
        latency_p99_ms=latency["p99"],
        latency_mean_ms=latency["mean"],
        satisfaction=satisfaction,
        satisfaction_sample_size=sample,
        num_queries=len(items),
    )

    print("\n=== RAG Evaluation Matrix ===")
    print(json.dumps(matrix.to_dict(), indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(matrix.to_dict(), f, indent=2)
        logger.info("Wrote matrix to %s", args.output)

    if args.save_baseline:
        from green_gov_rag.eval.regression import save_baseline

        save_baseline(matrix, args.save_baseline)
        logger.info("Saved baseline to %s", args.save_baseline)

    exit_code = 0
    if args.baseline:
        from green_gov_rag.eval.regression import compare_to_baseline, load_baseline

        baseline = load_baseline(args.baseline)
        if baseline is None:
            logger.warning("Baseline %s not found; skipping regression check", args.baseline)
        else:
            result = compare_to_baseline(matrix, baseline)
            print("\n=== Regression Check ===")
            print(json.dumps(result.to_dict(), indent=2))
            if not result.passed:
                logger.error("Regression gate FAILED")
                exit_code = 1
            else:
                logger.info("Regression gate passed")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
