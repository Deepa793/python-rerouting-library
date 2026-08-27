#!/usr/bin/env python3
"""
evaluate_router.py

Evaluate a simple/complex query router against router_queries_50.csv.

Metrics:
- Accuracy
- Precision (complex as positive class)
- Recall
- F1
- Confusion matrix
- CPU latency per router call
- Wall-clock latency per router call

IMPORTANT:
This script cannot access ChatGPT's internal production router.
Replace `route_query()` with your own local CPU router implementation,
or import/call your router from that function.

Expected router output:
    "simple"
or
    "complex"

Usage:
    python evaluate_router.py --csv router_queries_50.csv

Optional:
    python evaluate_router.py --csv router_queries_50.csv --repeats 20 --warmup 5
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


VALID_LABELS = {"simple", "complex"}


# ---------------------------------------------------------------------------
# ROUTER UNDER TEST
# ---------------------------------------------------------------------------

def route_query(query: str) -> str:
    """
    Replace this function with the router you want to benchmark.

    Example:
        from my_router import classify
        return classify(query)

    The function must return exactly:
        "simple" or "complex"
    """

    # Demo-only baseline so the evaluation harness is runnable immediately.
    # This is NOT ChatGPT's internal router.
    complex_markers = (
        "design ",
        "analyze ",
        "architecture",
        "strategy",
        "compare ",
        "evaluate ",
        "investigate ",
        "redesign ",
        "determine whether",
        "tradeoff",
        "trade-off",
        "migration",
        "root cause",
        "workflow",
        "pipeline",
        "multi-tenant",
        "disaster-recovery",
        "fraud-detection",
        "observability",
        "prioritized",
        "recommend an architecture",
        "order of interventions",
    )

    q = query.lower()

    if any(marker in q for marker in complex_markers):
        return "complex"

    return "simple"


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class Example:
    row_id: int
    label: str
    query: str


@dataclass
class Prediction:
    row_id: int
    expected: str
    predicted: str
    cpu_ms: float
    wall_ms: float


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_suite(csv_path: Path) -> list[Example]:
    rows: list[Example] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"id", "label", "query"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}"
            )

        for raw in reader:
            label = raw["label"].strip().lower()

            if label not in VALID_LABELS:
                raise ValueError(
                    f"Invalid label on row {raw['id']}: {label!r}"
                )

            rows.append(
                Example(
                    row_id=int(raw["id"]),
                    label=label,
                    query=raw["query"],
                )
            )

    return rows


# ---------------------------------------------------------------------------
# TIMING
# ---------------------------------------------------------------------------

def timed_route(
    router: Callable[[str], str],
    query: str,
    repeats: int,
) -> tuple[str, float, float]:
    """
    Returns:
        predicted_label,
        median CPU milliseconds,
        median wall milliseconds

    CPU time uses time.process_time_ns(), which measures process CPU time.
    Wall time uses time.perf_counter_ns().
    """

    predictions: list[str] = []
    cpu_samples_ms: list[float] = []
    wall_samples_ms: list[float] = []

    for _ in range(repeats):
        cpu_start = time.process_time_ns()
        wall_start = time.perf_counter_ns()

        predicted = router(query)

        wall_end = time.perf_counter_ns()
        cpu_end = time.process_time_ns()

        predicted = predicted.strip().lower()

        if predicted not in VALID_LABELS:
            raise ValueError(
                f"Router returned invalid label {predicted!r} "
                f"for query: {query!r}"
            )

        predictions.append(predicted)
        cpu_samples_ms.append((cpu_end - cpu_start) / 1_000_000.0)
        wall_samples_ms.append((wall_end - wall_start) / 1_000_000.0)

    # Ensure repeated runs are deterministic for classification.
    if len(set(predictions)) != 1:
        raise RuntimeError(
            f"Router was nondeterministic for query {query!r}: {predictions}"
        )

    return (
        predictions[0],
        statistics.median(cpu_samples_ms),
        statistics.median(wall_samples_ms),
    )


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(predictions: Iterable[Prediction]) -> dict:
    preds = list(predictions)

    tp = sum(
        1
        for p in preds
        if p.expected == "complex" and p.predicted == "complex"
    )
    fp = sum(
        1
        for p in preds
        if p.expected == "simple" and p.predicted == "complex"
    )
    tn = sum(
        1
        for p in preds
        if p.expected == "simple" and p.predicted == "simple"
    )
    fn = sum(
        1
        for p in preds
        if p.expected == "complex" and p.predicted == "simple"
    )

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    accuracy = safe_div(tp + tn, len(preds))
    f1 = safe_div(
        2 * precision * recall,
        precision + recall,
    )

    cpu = [p.cpu_ms for p in preds]
    wall = [p.wall_ms for p in preds]

    return {
        "count": len(preds),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cpu_mean_ms": statistics.mean(cpu) if cpu else 0.0,
        "cpu_median_ms": statistics.median(cpu) if cpu else 0.0,
        "cpu_p95_ms": percentile(cpu, 0.95),
        "cpu_max_ms": max(cpu) if cpu else 0.0,
        "wall_mean_ms": statistics.mean(wall) if wall else 0.0,
        "wall_median_ms": statistics.median(wall) if wall else 0.0,
        "wall_p95_ms": percentile(wall, 0.95),
        "wall_max_ms": max(wall) if wall else 0.0,
    }


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate(
    suite: list[Example],
    router: Callable[[str], str],
    repeats: int,
    warmup: int,
) -> list[Prediction]:

    if warmup > 0 and suite:
        warmup_query = suite[0].query
        for _ in range(warmup):
            router(warmup_query)

    results: list[Prediction] = []

    for example in suite:
        predicted, cpu_ms, wall_ms = timed_route(
            router=router,
            query=example.query,
            repeats=repeats,
        )

        results.append(
            Prediction(
                row_id=example.row_id,
                expected=example.label,
                predicted=predicted,
                cpu_ms=cpu_ms,
                wall_ms=wall_ms,
            )
        )

    return results


def print_report(results: list[Prediction]) -> None:
    metrics = compute_metrics(results)

    print()
    print("=" * 72)
    print("ROUTER EVALUATION")
    print("=" * 72)
    print(f"Queries evaluated : {metrics['count']}")
    print()
    print("Classification")
    print("-" * 72)
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Precision(complex): {metrics['precision']:.4f}")
    print(f"Recall(complex)   : {metrics['recall']:.4f}")
    print(f"F1(complex)       : {metrics['f1']:.4f}")
    print()
    print("Confusion matrix")
    print("-" * 72)
    print(f"TP complex->complex : {metrics['tp']}")
    print(f"FP simple ->complex : {metrics['fp']}")
    print(f"TN simple ->simple  : {metrics['tn']}")
    print(f"FN complex->simple  : {metrics['fn']}")
    print()
    print("CPU latency per query")
    print("-" * 72)
    print(f"Mean   : {metrics['cpu_mean_ms']:.6f} ms")
    print(f"Median : {metrics['cpu_median_ms']:.6f} ms")
    print(f"P95    : {metrics['cpu_p95_ms']:.6f} ms")
    print(f"Max    : {metrics['cpu_max_ms']:.6f} ms")
    print()
    print("Wall-clock latency per query")
    print("-" * 72)
    print(f"Mean   : {metrics['wall_mean_ms']:.6f} ms")
    print(f"Median : {metrics['wall_median_ms']:.6f} ms")
    print(f"P95    : {metrics['wall_p95_ms']:.6f} ms")
    print(f"Max    : {metrics['wall_max_ms']:.6f} ms")

    mismatches = [
        p for p in results if p.expected != p.predicted
    ]

    print()
    print("Mismatches")
    print("-" * 72)

    if not mismatches:
        print("None")
    else:
        for p in mismatches:
            print(
                f"Row {p.row_id}: "
                f"expected={p.expected}, predicted={p.predicted}"
            )

    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate simple/complex query-router precision and CPU latency."
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("router_queries_50.csv"),
        help="Path to evaluation CSV.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of timed router calls per query. Default: 10",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Untimed warmup calls before benchmarking. Default: 5",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    if args.warmup < 0:
        parser.error("--warmup must be >= 0")

    suite = load_suite(args.csv)

    simple_count = sum(x.label == "simple" for x in suite)
    complex_count = sum(x.label == "complex" for x in suite)

    if len(suite) != 50:
        raise AssertionError(
            f"Expected 50 evaluation rows; found {len(suite)}"
        )

    if simple_count != 25 or complex_count != 25:
        raise AssertionError(
            f"Expected 25 simple / 25 complex; "
            f"found {simple_count} simple / {complex_count} complex"
        )

    results = evaluate(
        suite=suite,
        router=route_query,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    print_report(results)


if __name__ == "__main__":
    main()
