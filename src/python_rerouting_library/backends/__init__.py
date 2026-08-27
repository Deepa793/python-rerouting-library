from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .cloud import CloudBackend
    from .local_llama import LocalLlamaBackend


__all__ = [
    "LocalLlamaBackend",
    "CloudBackend",
]


def __getattr__(name: str) -> Any:
    if name == "CloudBackend":
        from .cloud import CloudBackend

        return CloudBackend

    if name == "LocalLlamaBackend":
        from .local_llama import LocalLlamaBackend

        return LocalLlamaBackend

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
