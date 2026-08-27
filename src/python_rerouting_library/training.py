from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from .router import DEFAULT_EMBEDDING_MODEL


def load_training_data(csv_path: Path) -> tuple[list[str], list[str]]:
    queries: list[str] = []
    labels: list[str] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"label", "query"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "CSV must contain 'label' and 'query' columns."
            )

        for row in reader:
            label = row["label"].strip().lower()
            query = row["query"].strip()

            if label not in {"simple", "complex"}:
                raise ValueError(
                    f"Invalid label: {label!r}"
                )

            if not query:
                raise ValueError("Encountered an empty query.")

            labels.append(label)
            queries.append(query)

    if not queries:
        raise ValueError("Training CSV is empty.")

    return queries, labels


def train_router(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    threshold: float = 0.5,
) -> Path:
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    queries, labels = load_training_data(csv_path)

    print(
        f"Loaded {len(queries)} training examples: "
        f"{labels.count('simple')} simple / "
        f"{labels.count('complex')} complex"
    )

    print(f"Loading embedding model: {embedding_model_name}")

    encoder = SentenceTransformer(
        embedding_model_name,
        device="cpu",
    )

    print("Generating embeddings...")

    embeddings = encoder.encode(
        queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Training Logistic Regression classifier...")

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    classifier.fit(
        embeddings,
        labels,
    )

    artifact = {
        "classifier": classifier,
        "embedding_model_name": embedding_model_name,
        "threshold": float(threshold),
        "training_examples": len(queries),
        "classes": list(classifier.classes_),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        output_path,
    )

    print(f"Saved router classifier to: {output_path.resolve()}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the MiniLM + Logistic Regression "
            "simple/complex router."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/router_classifier.joblib"
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        parser.error(
            "--threshold must be between 0 and 1."
        )

    train_router(
        csv_path=args.csv,
        output_path=args.output,
        embedding_model_name=args.model,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
