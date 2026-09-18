from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from .exceptions import (
    CloudBackendError,
    DispatchError,
    LocalBackendError,
)
from .privacy import PrivacyDetector
from .router import RouteDecision, Router


class Backend(Protocol):
    name: str

    def generate(
        self,
        query: str,
    ) -> str:
        ...


@dataclass(frozen=True)
class DispatchResult:
    text: str
    route: RouteDecision
    backend_name: str
    fallback_used: bool = False
    privacy_categories: tuple[str, ...] = ()


class Dispatcher:
    def __init__(
        self,
        *,
        router: Router,
        simple_backend: Backend,
        complex_backend: Backend,
        privacy_detector: (
            PrivacyDetector | None
        ) = None,
    ) -> None:
        self.router = router

        self.simple_backend = (
            simple_backend
        )

        self.complex_backend = (
            complex_backend
        )

        self.privacy_detector = (
            privacy_detector
            or PrivacyDetector()
        )

    def run(
        self,
        query: str,
    ) -> DispatchResult:
        privacy_start = perf_counter()

        privacy_decision = (
            self.privacy_detector.detect(
                query
            )
        )

        privacy_latency_ms = (
            perf_counter()
            - privacy_start
        ) * 1000.0

        # PRIVACY OVERRIDE:
        # local execution only.
        #
        # Cloud fallback is intentionally
        # prohibited for privacy-sensitive
        # queries.
        if privacy_decision.is_sensitive:
            decision = RouteDecision(
                label="privacy_override",
                confidence=None,
                complex_probability=None,
                latency_ms=privacy_latency_ms,
            )

            try:
                text = (
                    self.simple_backend.generate(
                        query
                    )
                )

                return DispatchResult(
                    text=text,
                    route=decision,
                    backend_name=(
                        self.simple_backend.name
                    ),
                    fallback_used=False,
                    privacy_categories=(
                        privacy_decision.categories
                    ),
                )

            except LocalBackendError as error:
                raise DispatchError(
                    "Privacy override requires "
                    "local processing, but the "
                    "local backend failed. "
                    "Cloud fallback was not "
                    "attempted."
                ) from error

        # Clean queries continue to the
        # semantic complexity router.
        decision = self.router.route(query)

        # SIMPLE:
        # local first, cloud fallback.
        if decision.label == "simple":
            try:
                text = (
                    self.simple_backend.generate(
                        query
                    )
                )

                return DispatchResult(
                    text=text,
                    route=decision,
                    backend_name=(
                        self.simple_backend.name
                    ),
                    fallback_used=False,
                )

            except LocalBackendError:
                try:
                    text = (
                        self.complex_backend.generate(
                            query
                        )
                    )

                    return DispatchResult(
                        text=text,
                        route=decision,
                        backend_name=(
                            self.complex_backend.name
                        ),
                        fallback_used=True,
                    )

                except CloudBackendError as error:
                    raise DispatchError(
                        "Local backend failed "
                        "and cloud fallback "
                        "also failed."
                    ) from error

        # COMPLEX or UNCERTAIN:
        # cloud only.
        if decision.label in {
            "complex",
            "uncertain",
        }:
            try:
                text = (
                    self.complex_backend.generate(
                        query
                    )
                )

                return DispatchResult(
                    text=text,
                    route=decision,
                    backend_name=(
                        self.complex_backend.name
                    ),
                    fallback_used=False,
                )

            except CloudBackendError as error:
                raise DispatchError(
                    "Cloud backend failed for "
                    "a query requiring cloud "
                    "routing. Local fallback "
                    "was not attempted."
                ) from error

        raise DispatchError(
            "Unsupported route label: "
            f"{decision.label!r}"
        )
