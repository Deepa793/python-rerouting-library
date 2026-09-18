from pathlib import Path

import pytest

from python_rerouting_library.config import (
    Settings,
)
from python_rerouting_library.router import (
    Router,
)


def configure_required_environment(
    monkeypatch,
    tmp_path: Path,
) -> Path:
    llama_model = (
        tmp_path / "model.gguf"
    )

    llama_model.write_bytes(
        b"mock-model"
    )

    monkeypatch.setenv(
        "LLAMA_MODEL_PATH",
        str(llama_model),
    )

    monkeypatch.setenv(
        "CLOUD_API_KEY",
        "mock-api-key",
    )

    return llama_model


def test_settings_requires_classifier_path(
    monkeypatch,
    tmp_path,
):
    configure_required_environment(
        monkeypatch,
        tmp_path,
    )

    monkeypatch.delenv(
        "ROUTER_CLASSIFIER_PATH",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="ROUTER_CLASSIFIER_PATH",
    ):
        Settings.from_env()


def test_settings_rejects_missing_classifier_file(
    monkeypatch,
    tmp_path,
):
    configure_required_environment(
        monkeypatch,
        tmp_path,
    )

    missing_classifier = (
        tmp_path
        / "missing_classifier.joblib"
    )

    monkeypatch.setenv(
        "ROUTER_CLASSIFIER_PATH",
        str(missing_classifier),
    )

    with pytest.raises(
        FileNotFoundError,
        match="does not bundle a default classifier",
    ):
        Settings.from_env()


def test_router_missing_classifier_has_helpful_error(
    tmp_path,
):
    missing_classifier = (
        tmp_path
        / "missing_classifier.joblib"
    )

    with pytest.raises(
        FileNotFoundError,
        match="does not ship",
    ):
        Router(
            classifier_path=missing_classifier
        )