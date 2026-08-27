from python_rerouting_library import Router, Dispatcher
from python_rerouting_library.backends import LocalLlamaBackend


class FakeCloudBackend:
    name = "fake-cloud"

    def generate(self, query: str) -> str:
        return "FAKE CLOUD RESPONSE"


router = Router(
    "artifacts/router_classifier.joblib"
)

router.warmup()

local = LocalLlamaBackend(
    model_path=r"C:\models\llama3.2\model.gguf",
    temperature=0.0,
    max_tokens=128,
)

cloud = FakeCloudBackend()

dispatcher = Dispatcher(
    router=router,
    simple_backend=local,
    complex_backend=cloud,
)

query = "What does HTTP 404 mean?"

result = dispatcher.run(query)

print("Query:", query)
print("Route:", result.route.label)
print("Confidence:", round(result.route.confidence, 4))
print("Backend:", result.backend_name)
print("Response:")
print(result.text)