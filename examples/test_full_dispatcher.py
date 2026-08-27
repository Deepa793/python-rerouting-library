from python_rerouting_library import Router, Dispatcher
from python_rerouting_library.backends import (
    LocalLlamaBackend,
    CloudBackend,
)
from python_rerouting_library.config import Settings


settings = Settings.from_env()

router = Router(
    settings.router_classifier_path,
    simple_threshold=settings.router_simple_threshold,
    complex_threshold=settings.router_complex_threshold,
)

router.warmup()

local = LocalLlamaBackend(
    model_path=settings.llama_model_path,
    temperature=0.0,
    max_tokens=settings.local_max_tokens,
)

cloud = CloudBackend(
    base_url=settings.cloud_base_url,
    model=settings.cloud_model,
    api_key=settings.cloud_api_key,
    max_tokens=settings.cloud_max_tokens,
    temperature=0.0,
)

dispatcher = Dispatcher(
    router=router,
    simple_backend=local,
    complex_backend=cloud,
)

queries = [
    "What does HTTP 404 mean?",
    (
        "Design a multi-region database architecture with automatic "
        "failover, data consistency, disaster recovery, and low latency."
    ),
]

for query in queries:
    print("\n" + "=" * 70)
    print("Query:", query)

    result = dispatcher.run(query)

    print("Route:", result.route.label)
    print("Confidence:", round(result.route.confidence, 4))
    print(
        "P(complex):",
        round(result.route.complex_probability, 4),
    )
    print(
        "Router latency (ms):",
        round(result.route.latency_ms, 3),
    )
    print("Backend:", result.backend_name)
    print("Response:")
    print(result.text)