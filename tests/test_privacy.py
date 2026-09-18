from python_rerouting_library.privacy import (
    PrivacyDetector,
)


def make_detector():
    return PrivacyDetector()


def test_detects_email():
    detector = make_detector()

    decision = detector.detect(
        "Contact alice@example.com"
    )

    assert decision.is_sensitive is True
    assert "email" in decision.categories


def test_detects_phone_number():
    detector = make_detector()

    decision = detector.detect(
        "Call me at 303-555-0123"
    )

    assert decision.is_sensitive is True
    assert "phone" in decision.categories


def test_detects_ssn():
    detector = make_detector()

    decision = detector.detect(
        "SSN is 123-45-6789"
    )

    assert decision.is_sensitive is True
    assert "ssn" in decision.categories


def test_detects_openai_style_api_key():
    detector = make_detector()

    mock_key = "sk-" + ("A" * 24)

    decision = detector.detect(
        f"Use this key: {mock_key}"
    )

    assert decision.is_sensitive is True
    assert "api_key" in decision.categories


def test_detects_github_style_token():
    detector = make_detector()

    mock_token = "ghp_" + ("B" * 24)

    decision = detector.detect(
        f"Token: {mock_token}"
    )

    assert decision.is_sensitive is True
    assert "api_key" in decision.categories


def test_detects_valid_credit_card():
    detector = make_detector()

    decision = detector.detect(
        "Card: 4111 1111 1111 1111"
    )

    assert decision.is_sensitive is True
    assert (
        "credit_card"
        in decision.categories
    )


def test_rejects_invalid_credit_card():
    detector = make_detector()

    decision = detector.detect(
        "Number: 1234 5678 9012 3456"
    )

    assert "credit_card" not in (
        decision.categories
    )


def test_clean_query_is_not_sensitive():
    detector = make_detector()

    decision = detector.detect(
        "Explain Python decorators."
    )

    assert decision.is_sensitive is False
    assert decision.categories == ()


def test_complex_query_with_email_is_sensitive():
    detector = make_detector()

    decision = detector.detect(
        "Design a multi-region database "
        "architecture for alice@example.com "
        "with automatic failover."
    )

    assert decision.is_sensitive is True
    assert "email" in decision.categories


def test_detects_multiple_categories():
    detector = make_detector()

    decision = detector.detect(
        "Email alice@example.com and "
        "use SSN 123-45-6789."
    )

    assert decision.is_sensitive is True

    assert set(
        decision.categories
    ) == {
        "email",
        "ssn",
    }