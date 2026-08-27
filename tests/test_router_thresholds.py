import pytest

from python_rerouting_library.router import Router


class FakeEmbeddingModel:
    def encode(
        self,
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        return [[0.0]]


def make_router_with_probability(probability: float) -> Router:
    """
    Create a Router instance without loading:
    - the classifier artifact
    - MiniLM
    - Llama
    - any cloud backend

    Only the routing threshold logic is exercised.
    """
    router = Router.__new__(Router)

    router.simple_threshold = 0.40
    router.complex_threshold = 0.60

    fake_model = FakeEmbeddingModel()

    router._get_embedding_model = lambda: fake_model
    router._get_complex_probability = (
        lambda embedding: probability
    )

    return router


def test_probability_below_simple_threshold_is_simple():
    router = make_router_with_probability(0.39)

    decision = router.route("test query")

    assert decision.label == "simple"
    assert decision.complex_probability == pytest.approx(0.39)
    assert decision.confidence == pytest.approx(0.61)


def test_probability_at_simple_threshold_is_uncertain():
    router = make_router_with_probability(0.40)

    decision = router.route("test query")

    assert decision.label == "uncertain"
    assert decision.complex_probability == pytest.approx(0.40)
    assert decision.confidence == pytest.approx(0.20)


def test_probability_at_midpoint_is_uncertain():
    router = make_router_with_probability(0.50)

    decision = router.route("test query")

    assert decision.label == "uncertain"
    assert decision.complex_probability == pytest.approx(0.50)
    assert decision.confidence == pytest.approx(0.0)


def test_probability_at_complex_threshold_is_uncertain():
    router = make_router_with_probability(0.60)

    decision = router.route("test query")

    assert decision.label == "uncertain"
    assert decision.complex_probability == pytest.approx(0.60)
    assert decision.confidence == pytest.approx(0.20)


def test_probability_above_complex_threshold_is_complex():
    router = make_router_with_probability(0.61)

    decision = router.route("test query")

    assert decision.label == "complex"
    assert decision.complex_probability == pytest.approx(0.61)
    assert decision.confidence == pytest.approx(0.61)