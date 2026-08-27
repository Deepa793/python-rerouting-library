from __future__ import annotations

from openai import OpenAI

from python_rerouting_library.exceptions import CloudBackendError


class CloudBackend:
    name = "cloud-api"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, query: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content

            if not content:
                raise CloudBackendError(
                    "Cloud provider returned no text."
                )

            return content

        except CloudBackendError:
            raise

        except Exception as exc:
            raise CloudBackendError(
                "Cloud API request failed."
            ) from exc