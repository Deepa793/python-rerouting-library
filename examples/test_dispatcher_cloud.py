import os

from python_rerouting_library import Router, Dispatcher
from python_rerouting_library.backends import (
    LocalLlamaBackend,
    CloudBackend,
)

router = Router(
    "artifacts/router_classifier.joblib"
)

router.warmup()

local = LocalLlamaBackend(
    model_path=r"C:\models\llama3.2\model.gguf",
    temperature=0.0,
    max_tokens=128,
)

cloud = CloudBackend(
    base_url="https://api.openai.com/v1",
    model="gpt-5.4-nano",
    api_key=os.environ["CLOUD_API_KEY"],
    max_tokens=256,
    temperature=0.0,
)

dispatcher = Dispatcher(
    router=router,
    simple_backend=local,
    complex_backend=cloud,
)

query = (
    "Design a multi-region database architecture with automatic failover, "
    "data consistency, disaster recovery, and low latency."
)

result = dispatcher.run(query)

print("Query:", query)
print("Route:", result.route.label)
print("Confidence:", round(result.route.confidence, 4))
print("Backend:", result.backend_name)
print("Response:")
print(result.text)