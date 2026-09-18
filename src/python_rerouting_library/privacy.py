from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacyDecision:
    is_sensitive: bool
    categories: tuple[str, ...]


class PrivacyDetector:
    """
    Detect common sensitive-data patterns before cloud routing.

    The detector returns only category names. It never returns
    or logs the matched sensitive value.
    """

    _EMAIL_PATTERNS = (
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@"
            r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
    )

    _PHONE_PATTERNS = (
        re.compile(
            r"(?<!\d)"
            r"(?:\+?1[\s.-]?)?"
            r"(?:\(\d{3}\)|\d{3})"
            r"[\s.-]?\d{3}"
            r"[\s.-]?\d{4}"
            r"(?!\d)"
        ),
    )

    _SSN_PATTERNS = (
        re.compile(
            r"\b\d{3}-\d{2}-\d{4}\b"
        ),
    )

    _API_KEY_PATTERNS = (
        # OpenAI-style secret keys.
        re.compile(
            r"\bsk-[A-Za-z0-9_-]{20,}\b"
        ),

        # GitHub classic-style tokens.
        re.compile(
            r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"
        ),

        # GitHub fine-grained tokens.
        re.compile(
            r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"
        ),

        # Google API-key style.
        re.compile(
            r"\bAIza[A-Za-z0-9_-]{35}\b"
        ),

        # AWS access-key identifiers.
        re.compile(
            r"\bAKIA[0-9A-Z]{16}\b"
        ),
    )

    _CREDIT_CARD_CANDIDATE = re.compile(
        r"(?<!\d)"
        r"(?:\d[ -]*?){12,18}\d"
        r"(?!\d)"
    )

    def detect(self, text: str) -> PrivacyDecision:
        if not isinstance(text, str):
            raise TypeError(
                "PrivacyDetector input must be a string."
            )

        categories: list[str] = []

        if self._matches(
            text,
            self._EMAIL_PATTERNS,
        ):
            categories.append("email")

        if self._matches(
            text,
            self._PHONE_PATTERNS,
        ):
            categories.append("phone")

        if self._matches(
            text,
            self._SSN_PATTERNS,
        ):
            categories.append("ssn")

        if self._matches(
            text,
            self._API_KEY_PATTERNS,
        ):
            categories.append("api_key")

        if self._contains_credit_card(text):
            categories.append("credit_card")

        return PrivacyDecision(
            is_sensitive=bool(categories),
            categories=tuple(categories),
        )

    @staticmethod
    def _matches(
        text: str,
        patterns: tuple[re.Pattern[str], ...],
    ) -> bool:
        return any(
            pattern.search(text)
            for pattern in patterns
        )

    def _contains_credit_card(
        self,
        text: str,
    ) -> bool:
        for match in self._CREDIT_CARD_CANDIDATE.finditer(
            text
        ):
            digits = re.sub(
                r"\D",
                "",
                match.group(0),
            )

            if not 13 <= len(digits) <= 19:
                continue

            if len(set(digits)) == 1:
                continue

            if self._passes_luhn(digits):
                return True

        return False

    @staticmethod
    def _passes_luhn(number: str) -> bool:
        digits = [
            int(character)
            for character in number
        ]

        checksum = 0
        parity = len(digits) % 2

        for index, digit in enumerate(digits):
            if index % 2 == parity:
                digit *= 2

                if digit > 9:
                    digit -= 9

            checksum += digit

        return checksum % 10 == 0