#!/usr/bin/env python3
"""
evaluate_embedding_router.py

Local semantic router benchmark:

    Query
      -> sentence-transformers/all-MiniLM-L6-v2 (CPU)
      -> 384-dimensional normalized embedding
      -> Logistic Regression
      -> simple / complex

Evaluation:
- 5-fold Stratified Cross-Validation
- Every query is tested out-of-fold
- Pretrained embedding model is fixed; labels are used only by Logistic Regression
- Reports classification metrics and per-query inference latency
- Inference latency INCLUDES:
      embedding generation + Logistic Regression probability
- Inference latency EXCLUDES:
      model download + initial model loading

Requirements:
    python -m pip install -U sentence-transformers scikit-learn

Usage:
    python evaluate_embedding_router.py --csv router_queries_50.csv

More stable timing:
    python evaluate_embedding_router.py --csv router_queries_50.csv --repeats 20 --cpu-batch-size 20
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import StratifiedKFold
except ImportError as exc:
    raise SystemExit(
        "\nRequired packages are missing.\n\n"
        "Run:\n"
        "  python -m pip install -U sentence-transformers scikit-learn\n"
    ) from exc


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
POSITIVE_LABEL = "complex"
VALID_LABELS = {"simple", "complex"}


def load_data(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"id", "label", "query"}
        actual = set(reader.fieldnames or [])

        if not required.issubset(actual):
            raise ValueError(
                f"CSV must contain columns: {sorted(required)}"
            )

        for row in reader:
            label = row["label"].strip().lower()

            if label not in VALID_LABELS:
                raise ValueError(
                    f"Invalid label on row {row['id']}: {label!r}"
                )

            rows.append(
                {
                    "id": int(row["id"]),
                    "label": label,
                    "query": row["query"].strip(),
                }
            )

    if not rows:
        raise ValueError("CSV contains no rows.")

    return rows


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)

    if not values:
        return 0.0

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low

    return (
        values[low] * (1 - fraction)
        + values[high] * fraction
    )


def make_classifier() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )


def encode_queries(
    model: SentenceTransformer,
    queries: list[str],
) -> np.ndarray:
    """
    Encode queries using the pretrained semantic model.

    This is used to create training embeddings for cross-validation.
    """
    return model.encode(
        queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def route_once(
    model: SentenceTransformer,
    classifier: LogisticRegression,
    query: str,
) -> tuple[str, float]:
    """
    Full routing path for ONE query.

    Latency includes:
        query -> embedding -> Logistic Regression probability
    """
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    classes = list(classifier.classes_)
    complex_idx = classes.index(POSITIVE_LABEL)

    probability = float(
        classifier.predict_proba(embedding)[0][complex_idx]
    )

    predicted = (
        "complex"
        if probability >= 0.5
        else "simple"
    )

    return predicted, probability


def timed_route(
    model: SentenceTransformer,
    classifier: LogisticRegression,
    query: str,
    repeats: int,
    cpu_batch_size: int,
) -> tuple[str, float, float, float]:
    """
    Returns:
        predicted label,
        complex probability,
        CPU milliseconds per query,
        median wall milliseconds per query

    CPU timing is batched to avoid Windows process-time timer
    resolution issues on fast operations.
    """

    wall_samples = []
    predictions = []
    probabilities = []

    # Wall-clock measurement.
    for _ in range(repeats):
        w0 = time.perf_counter_ns()

        pred, prob = route_once(
            model,
            classifier,
            query,
        )

        w1 = time.perf_counter_ns()

        predictions.append(pred)
        probabilities.append(prob)
        wall_samples.append(
            (w1 - w0) / 1_000_000.0
        )

    if len(set(predictions)) != 1:
        raise RuntimeError(
            f"Nondeterministic predictions for: {query!r}"
        )

    # CPU timing over a batch.
    c0 = time.process_time_ns()

    for _ in range(cpu_batch_size):
        route_once(
            model,
            classifier,
            query,
        )

    c1 = time.process_time_ns()

    total_cpu_ms = (
        c1 - c0
    ) / 1_000_000.0

    cpu_ms_per_query = (
        total_cpu_ms / cpu_batch_size
    )

    return (
        predictions[0],
        statistics.median(probabilities),
        cpu_ms_per_query,
        statistics.median(wall_samples),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a local MiniLM embedding + Logistic Regression "
            "simple/complex query router."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("router_queries_50.csv"),
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help=(
            "Per-query wall-clock timing repetitions. "
            "Default: 10"
        ),
    )

    parser.add_argument(
        "--cpu-batch-size",
        type=int,
        default=20,
        help=(
            "Repeated full routes used for CPU timing. "
            "Embedding inference is much slower than TF-IDF, "
            "so the default is intentionally smaller. Default: 20"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "embedding_router_predictions.csv"
        ),
    )

    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be >= 2")

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    if args.cpu_batch_size < 1:
        parser.error("--cpu-batch-size must be >= 1")

    rows = load_data(args.csv)

    queries = [r["query"] for r in rows]
    labels = [r["label"] for r in rows]

    simple_count = labels.count("simple")
    complex_count = labels.count("complex")

    print(
        f"Loaded {len(rows)} rows: "
        f"{simple_count} simple / "
        f"{complex_count} complex"
    )

    print()
    print("Loading local embedding model:")
    print(f"  {MODEL_NAME}")
    print("Device:")
    print("  CPU")
    print()
    print(
        "Note: on the first run, model files may be "
        "downloaded and cached locally."
    )

    # Explicitly force CPU execution.
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print()
    print("Model loaded.")

    # Warm up model so one-time initialization is excluded
    # from latency measurements.
    print("Warming up embedding model...")
    model.encode(
        ["warmup query"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print("Warmup complete.")

    # Compute fixed pretrained embeddings once for model training.
    # This does not use labels and does not leak class targets.
    print("Generating dataset embeddings...")
    embeddings = encode_queries(
        model,
        queries,
    )

    print(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=42,
    )

    results = []
    training_wall_ms = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            embeddings,
            labels,
        ),
        start=1,
    ):

        classifier = make_classifier()

        X_train = embeddings[train_idx]
        y_train = [
            labels[i]
            for i in train_idx
        ]

        w0 = time.perf_counter_ns()

        classifier.fit(
            X_train,
            y_train,
        )

        w1 = time.perf_counter_ns()

        training_wall_ms.append(
            (w1 - w0) / 1_000_000.0
        )

        # Warm the exact classifier path for this fold.
        if len(test_idx):
            route_once(
                model,
                classifier,
                queries[test_idx[0]],
            )

        for i in test_idx:
            (
                pred,
                prob,
                cpu_ms,
                wall_ms,
            ) = timed_route(
                model=model,
                classifier=classifier,
                query=queries[i],
                repeats=args.repeats,
                cpu_batch_size=args.cpu_batch_size,
            )

            results.append(
                {
                    "id": rows[i]["id"],
                    "query": queries[i],
                    "expected": labels[i],
                    "predicted": pred,
                    "complex_probability": prob,
                    "fold": fold,
                    "cpu_ms": cpu_ms,
                    "wall_ms": wall_ms,
                }
            )

    results.sort(
        key=lambda x: x["id"]
    )

    y_true = [
        r["expected"]
        for r in results
    ]

    y_pred = [
        r["predicted"]
        for r in results
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="complex",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="complex",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="complex",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "simple",
            "complex",
        ],
    )

    tn, fp, fn, tp = cm.ravel()

    cpu = [
        r["cpu_ms"]
        for r in results
    ]

    wall = [
        r["wall_ms"]
        for r in results
    ]

    print()
    print("=" * 74)
    print(
        "MINILM EMBEDDING + LOGISTIC REGRESSION "
        "— 5-FOLD CROSS-VALIDATION"
    )
    print("=" * 74)

    print(
        f"Accuracy             : "
        f"{accuracy:.4f}"
    )
    print(
        f"Precision (complex)  : "
        f"{precision:.4f}"
    )
    print(
        f"Recall (complex)     : "
        f"{recall:.4f}"
    )
    print(
        f"F1 (complex)         : "
        f"{f1:.4f}"
    )
    print(
        f"Macro precision      : "
        f"{macro_precision:.4f}"
    )
    print(
        f"Macro F1             : "
        f"{macro_f1:.4f}"
    )

    print()

    print(
        f"TN simple->simple    : {tn}"
    )
    print(
        f"FP simple->complex   : {fp}"
    )
    print(
        f"FN complex->simple   : {fn}"
    )
    print(
        f"TP complex->complex  : {tp}"
    )

    print()
    print(
        "CPU inference latency "
        "(embedding + classifier)"
    )

    print(
        f"Mean                 : "
        f"{statistics.mean(cpu):.6f} ms"
    )
    print(
        f"Median               : "
        f"{statistics.median(cpu):.6f} ms"
    )
    print(
        f"P95                  : "
        f"{percentile(cpu, 0.95):.6f} ms"
    )
    print(
        f"Max                  : "
        f"{max(cpu):.6f} ms"
    )

    print()
    print(
        "Wall-clock inference latency "
        "(embedding + classifier)"
    )

    print(
        f"Mean                 : "
        f"{statistics.mean(wall):.6f} ms"
    )
    print(
        f"Median               : "
        f"{statistics.median(wall):.6f} ms"
    )
    print(
        f"P95                  : "
        f"{percentile(wall, 0.95):.6f} ms"
    )
    print(
        f"Max                  : "
        f"{max(wall):.6f} ms"
    )

    print()
    print(
        f"Mean classifier training/fold : "
        f"{statistics.mean(training_wall_ms):.6f} ms"
    )

    mismatches = [
        r
        for r in results
        if r["expected"]
        != r["predicted"]
    ]

    print()
    print("Misclassified queries:")

    if not mismatches:
        print("None")
    else:
        for r in mismatches:
            print(
                f'Row {r["id"]}: '
                f'expected={r["expected"]}, '
                f'predicted={r["predicted"]}, '
                f'P(complex)='
                f'{r["complex_probability"]:.3f}'
            )

            print(
                "    "
                + r["query"]
            )

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "query",
                "expected_label",
                "predicted_label",
                "complex_probability",
                "correct",
                "fold",
                "cpu_ms",
                "wall_ms",
            ],
        )

        writer.writeheader()

        for r in results:
            writer.writerow(
                {
                    "id": r["id"],
                    "query": r["query"],
                    "expected_label":
                        r["expected"],
                    "predicted_label":
                        r["predicted"],
                    "complex_probability":
                        f'{r["complex_probability"]:.6f}',
                    "correct":
                        r["expected"]
                        == r["predicted"],
                    "fold": r["fold"],
                    "cpu_ms":
                        f'{r["cpu_ms"]:.6f}',
                    "wall_ms":
                        f'{r["wall_ms"]:.6f}',
                }
            )

    print()
    print(
        f"Saved predictions to: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
