from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class RouteDecision:
    label: str
    confidence: float
    complex_probability: float
    latency_ms: float


class Router:
    def __init__(
        self,
        classifier_path: str | Path,
        *,
        embedding_model_name: str | None = None,
        simple_threshold: float = 0.40,
        complex_threshold: float = 0.60,
        device: str = "cpu",
    ) -> None:
        self.classifier_path = Path(classifier_path)

        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"Router classifier not found: {self.classifier_path}"
            )

        if not 0.0 < simple_threshold < complex_threshold < 1.0:
            raise ValueError(
                "Thresholds must satisfy: "
                "0 < simple_threshold < complex_threshold < 1"
            )

        self.simple_threshold = float(simple_threshold)
        self.complex_threshold = float(complex_threshold)
        self.device = device

        artifact = joblib.load(self.classifier_path)

        # Support the persisted artifact produced by training.py.
        if isinstance(artifact, dict):
            if "classifier" not in artifact:
                raise ValueError(
                    "Router artifact does not contain a 'classifier'."
                )

            self.classifier = artifact["classifier"]

            artifact_model_name = artifact.get(
                "embedding_model_name",
                DEFAULT_EMBEDDING_MODEL,
            )
        else:
            # Allows a classifier-only artifact if needed.
            self.classifier = artifact
            artifact_model_name = DEFAULT_EMBEDDING_MODEL

        self.embedding_model_name = (
            embedding_model_name
            or artifact_model_name
            or DEFAULT_EMBEDDING_MODEL
        )

        self._embedding_model: SentenceTransformer | None = None

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
            )

        return self._embedding_model

    def warmup(self) -> None:
        """
        Load the embedding model and run one small embedding operation.

        This avoids paying the model-loading cost on the first real query.
        """
        model = self._get_embedding_model()

        model.encode(
            ["router warmup"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _get_complex_probability(self, embedding) -> float:
        probabilities = self.classifier.predict_proba(embedding)[0]

        classes = list(self.classifier.classes_)

        if "complex" not in classes:
            raise ValueError(
                "Classifier does not contain the 'complex' class."
            )

        complex_index = classes.index("complex")

        return float(probabilities[complex_index])

    def route(self, query: str) -> RouteDecision:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query must not be empty.")

        start = perf_counter()

        model = self._get_embedding_model()

        embedding = model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        probability = self._get_complex_probability(embedding)

        if probability < self.simple_threshold:
            label = "simple"
            confidence = 1.0 - probability

        elif probability > self.complex_threshold:
            label = "complex"
            confidence = probability

        else:
            label = "uncertain"

            # In the uncertainty band, confidence expresses how far
            # the classifier is from the neutral 0.50 decision point.
            # At exactly 0.50, routing confidence is 0.
            confidence = abs(probability - 0.5) * 2.0

        latency_ms = (perf_counter() - start) * 1000.0

        return RouteDecision(
            label=label,
            confidence=confidence,
            complex_probability=probability,
            latency_ms=latency_ms,
        )