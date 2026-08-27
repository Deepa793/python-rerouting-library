from __future__ import annotations

from pathlib import Path

from llama_cpp import Llama

from python_rerouting_library.exceptions import LocalBackendError


class LocalLlamaBackend:
    name = "local-llama"

    def __init__(
        self,
        *,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Local Llama model was not found: {self.model_path}"
            )

        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
            )

        except Exception as exc:
            raise LocalBackendError(
                "Failed to load the local Llama model."
            ) from exc

    def generate(self, query: str) -> str:
        try:
            result = self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            content = result["choices"][0]["message"]["content"]

            if not content:
                raise LocalBackendError(
                    "Local Llama returned no text."
                )

            return content

        except LocalBackendError:
            raise

        except Exception as exc:
            raise LocalBackendError(
                "Local Llama generation failed."
            ) from exc
        return result["choices"][0]["message"]["content"]
