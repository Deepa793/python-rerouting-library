from python_rerouting_library.router import RouteDecision


def test_route_decision_fields():
    decision = RouteDecision(
        label="simple",
        confidence=0.8,
        complex_probability=0.2,
        latency_ms=3.5,
    )

    assert decision.label == "simple"
    assert decision.confidence == 0.8
    assert decision.complex_probability == 0.2
    assert decision.latency_ms == 3.5
