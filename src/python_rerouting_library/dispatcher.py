from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .exceptions import (
    CloudBackendError,
    DispatchError,
    LocalBackendError,
)
from .router import RouteDecision, Router


class Backend(Protocol):
    name: str

    def generate(self, query: str) -> str:
        ...


@dataclass(frozen=True)
class DispatchResult:
    text: str
    route: RouteDecision
    backend_name: str
    fallback_used: bool = False


class Dispatcher:
    def __init__(
        self,
        *,
        router: Router,
        simple_backend: Backend,
        complex_backend: Backend,
    ) -> None:
        self.router = router
        self.simple_backend = simple_backend
        self.complex_backend = complex_backend

    def run(self, query: str) -> DispatchResult:
        decision = self.router.route(query)

        # SIMPLE: local first, cloud fallback.
        if decision.label == "simple":
            try:
                text = self.simple_backend.generate(query)

                return DispatchResult(
                    text=text,
                    route=decision,
                    backend_name=self.simple_backend.name,
                    fallback_used=False,
                )

            except LocalBackendError:
                try:
                    text = self.complex_backend.generate(query)

                    return DispatchResult(
                        text=text,
                        route=decision,
                        backend_name=self.complex_backend.name,
                        fallback_used=True,
                    )

                except CloudBackendError as cloud_error:
                    raise DispatchError(
                        "Local backend failed and cloud "
                        "fallback also failed."
                    ) from cloud_error

        # COMPLEX or UNCERTAIN: cloud only.
        if decision.label in {"complex", "uncertain"}:
            try:
                text = self.complex_backend.generate(query)

                return DispatchResult(
                    text=text,
                    route=decision,
                    backend_name=self.complex_backend.name,
                    fallback_used=False,
                )

            except CloudBackendError as cloud_error:
                raise DispatchError(
                    "Cloud backend failed for a query requiring "
                    "cloud routing. Local fallback was not attempted."
                ) from cloud_error

        raise DispatchError(
            f"Unsupported route label: {decision.label!r}"
        )
