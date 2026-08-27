from python_rerouting_library import Router

router = Router(
    "artifacts/router_classifier.joblib"
)

router.warmup()

query = "What does HTTP 404 mean?"

decision = router.route(query)

print("Query:", query)
print("Label:", decision.label)
print("Confidence:", round(decision.confidence, 4))
print(
    "P(complex):",
    round(decision.complex_probability, 4),
)
print(
    "Latency (ms):",
    round(decision.latency_ms, 3),
)
