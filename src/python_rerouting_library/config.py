from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    router_classifier_path: Path
    llama_model_path: Path

    cloud_api_key: str
    cloud_base_url: str
    cloud_model: str

    local_max_tokens: int = 128
    cloud_max_tokens: int = 256

    router_simple_threshold: float = 0.40
    router_complex_threshold: float = 0.60

    @classmethod
    def from_env(cls) -> "Settings":
        llama_model_path = os.getenv(
            "LLAMA_MODEL_PATH"
        )

        cloud_api_key = os.getenv(
            "CLOUD_API_KEY"
        )

        router_classifier_path = os.getenv(
            "ROUTER_CLASSIFIER_PATH"
        )

        if not llama_model_path:
            raise RuntimeError(
                "LLAMA_MODEL_PATH environment "
                "variable is not set."
            )

        if not cloud_api_key:
            raise RuntimeError(
                "CLOUD_API_KEY environment "
                "variable is not set."
            )

        if not router_classifier_path:
            raise RuntimeError(
                "ROUTER_CLASSIFIER_PATH environment "
                "variable is not set. "
                "python-rerouting-library does not "
                "ship with a default complexity "
                "classifier. Train a classifier with "
                "'python -m "
                "python_rerouting_library.training' "
                "or provide your own compatible "
                "classifier artifact, then set "
                "ROUTER_CLASSIFIER_PATH to that file."
            )

        llama_path = Path(
            llama_model_path
        ).expanduser()

        if not llama_path.is_file():
            raise FileNotFoundError(
                "Local Llama model was not found: "
                f"{llama_path}"
            )

        router_path = Path(
            router_classifier_path
        ).expanduser()

        if not router_path.is_file():
            raise FileNotFoundError(
                "Router classifier was not found: "
                f"{router_path}. "
                "python-rerouting-library does not "
                "bundle a default classifier. "
                "Train one or provide a compatible "
                "classifier artifact and set "
                "ROUTER_CLASSIFIER_PATH to its path."
            )

        cloud_base_url = os.getenv(
            "CLOUD_BASE_URL",
            "https://api.openai.com/v1",
        )

        cloud_model = os.getenv(
            "CLOUD_MODEL",
            "gpt-5.4-nano",
        )

        return cls(
            router_classifier_path=router_path,
            llama_model_path=llama_path,
            cloud_api_key=cloud_api_key,
            cloud_base_url=cloud_base_url,
            cloud_model=cloud_model,
            local_max_tokens=int(
                os.getenv(
                    "LOCAL_MAX_TOKENS",
                    "128",
                )
            ),
            cloud_max_tokens=int(
                os.getenv(
                    "CLOUD_MAX_TOKENS",
                    "256",
                )
            ),
            router_simple_threshold=float(
                os.getenv(
                    "ROUTER_SIMPLE_THRESHOLD",
                    "0.40",
                )
            ),
            router_complex_threshold=float(
                os.getenv(
                    "ROUTER_COMPLEX_THRESHOLD",
                    "0.60",
                )
            ),
        )