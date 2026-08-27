#!/usr/bin/env python3
import argparse
import csv
import statistics
import time
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is not installed.\n"
        "Run: python -m pip install scikit-learn"
    ) from exc


def load_data(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "label", "query"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must contain id, label, query columns.")
        for r in reader:
            rows.append({
                "id": int(r["id"]),
                "label": r["label"].strip().lower(),
                "query": r["query"].strip(),
            })
    return rows


def build_router():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )),
    ])


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    x = (len(values) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    frac = x - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def timed_predict(model, query, repeats, cpu_batch_size=1000):
    """
    Measure one-query routing latency.

    Wall-clock latency:
        Timed per prediction and summarized with the median.

    CPU latency:
        Timed in a large batch and divided by the number of predictions.
        This avoids Windows process CPU timer resolution returning 0 ms
        for very fast individual predictions.
    """
    wall = []
    preds = []
    probs = []

    classes = list(model.named_steps["clf"].classes_)
    complex_idx = classes.index("complex")

    # Per-call wall-clock timing.
    for _ in range(repeats):
        w0 = time.perf_counter_ns()

        pred = model.predict([query])[0]
        prob = model.predict_proba([query])[0][complex_idx]

        w1 = time.perf_counter_ns()

        preds.append(pred)
        probs.append(float(prob))
        wall.append((w1 - w0) / 1_000_000)

    if len(set(preds)) != 1:
        raise RuntimeError("Router produced inconsistent predictions.")

    # Batched CPU timing of the full routing operation.
    c0 = time.process_time_ns()

    for _ in range(cpu_batch_size):
        model.predict([query])
        model.predict_proba([query])

    c1 = time.process_time_ns()

    total_cpu_ms = (c1 - c0) / 1_000_000
    cpu_ms_per_query = total_cpu_ms / cpu_batch_size

    return (
        preds[0],
        statistics.median(probs),
        cpu_ms_per_query,
        statistics.median(wall),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("router_queries_50.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--cpu-batch-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("tfidf_router_predictions.csv"))
    args = parser.parse_args()

    if args.cpu_batch_size < 1:
        parser.error("--cpu-batch-size must be at least 1")

    rows = load_data(args.csv)
    labels = [r["label"] for r in rows]
    queries = [r["query"] for r in rows]

    simple_count = labels.count("simple")
    complex_count = labels.count("complex")
    print(f"Loaded {len(rows)} rows: {simple_count} simple / {complex_count} complex")

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=42,
    )

    results = []
    train_cpu = []
    train_wall = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(queries, labels), start=1):
        model = build_router()

        train_q = [queries[i] for i in train_idx]
        train_y = [labels[i] for i in train_idx]

        c0 = time.process_time_ns()
        w0 = time.perf_counter_ns()
        model.fit(train_q, train_y)
        w1 = time.perf_counter_ns()
        c1 = time.process_time_ns()

        train_cpu.append((c1 - c0) / 1_000_000)
        train_wall.append((w1 - w0) / 1_000_000)

        if len(test_idx):
            model.predict([queries[test_idx[0]]])
            model.predict_proba([queries[test_idx[0]]])

        for i in test_idx:
            pred, prob, cpu_ms, wall_ms = timed_predict(
                model, queries[i], args.repeats, args.cpu_batch_size
            )
            results.append({
                "id": rows[i]["id"],
                "query": queries[i],
                "expected": labels[i],
                "predicted": pred,
                "complex_probability": prob,
                "fold": fold,
                "cpu_ms": cpu_ms,
                "wall_ms": wall_ms,
            })

    results.sort(key=lambda x: x["id"])

    y_true = [r["expected"] for r in results]
    y_pred = [r["predicted"] for r in results]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, pos_label="complex", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, pos_label="complex", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, pos_label="complex", zero_division=0
    )
    macro_precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=["simple", "complex"])
    tn, fp, fn, tp = cm.ravel()

    cpu = [r["cpu_ms"] for r in results]
    wall = [r["wall_ms"] for r in results]

    print("\n" + "=" * 70)
    print("TF-IDF + LOGISTIC REGRESSION — 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    print(f"Accuracy             : {accuracy:.4f}")
    print(f"Precision (complex)  : {precision:.4f}")
    print(f"Recall (complex)     : {recall:.4f}")
    print(f"F1 (complex)         : {f1:.4f}")
    print(f"Macro precision      : {macro_precision:.4f}")
    print(f"Macro F1             : {macro_f1:.4f}")
    print()
    print(f"TN simple->simple    : {tn}")
    print(f"FP simple->complex   : {fp}")
    print(f"FN complex->simple   : {fn}")
    print(f"TP complex->complex  : {tp}")
    print()
    print("CPU inference latency")
    print(f"Mean                 : {statistics.mean(cpu):.6f} ms")
    print(f"Median               : {statistics.median(cpu):.6f} ms")
    print(f"P95                  : {percentile(cpu, 0.95):.6f} ms")
    print(f"Max                  : {max(cpu):.6f} ms")
    print()
    print("Wall-clock inference latency")
    print(f"Mean                 : {statistics.mean(wall):.6f} ms")
    print(f"Median               : {statistics.median(wall):.6f} ms")
    print(f"P95                  : {percentile(wall, 0.95):.6f} ms")
    print(f"Max                  : {max(wall):.6f} ms")
    print()
    print(f"Mean CPU training/fold  : {statistics.mean(train_cpu):.6f} ms")
    print(f"Mean wall training/fold : {statistics.mean(train_wall):.6f} ms")

    mismatches = [r for r in results if r["expected"] != r["predicted"]]
    print("\nMisclassified queries:")
    if not mismatches:
        print("None")
    else:
        for r in mismatches:
            print(
                f'Row {r["id"]}: expected={r["expected"]}, '
                f'predicted={r["predicted"]}, '
                f'P(complex)={r["complex_probability"]:.3f}'
            )
            print("   ", r["query"])

    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "query", "expected_label", "predicted_label",
                "complex_probability", "correct", "fold",
                "cpu_ms", "wall_ms"
            ]
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "id": r["id"],
                "query": r["query"],
                "expected_label": r["expected"],
                "predicted_label": r["predicted"],
                "complex_probability": f'{r["complex_probability"]:.6f}',
                "correct": r["expected"] == r["predicted"],
                "fold": r["fold"],
                "cpu_ms": f'{r["cpu_ms"]:.6f}',
                "wall_ms": f'{r["wall_ms"]:.6f}',
            })

    print(f"\nSaved predictions to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
