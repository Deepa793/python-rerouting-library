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
    model_path=r"C:\models\llama3.2\model.gguf"
)

cloud = CloudBackend(
    base_url=os.environ["CLOUD_BASE_URL"],
    model=os.environ["CLOUD_MODEL"],
    api_key=os.environ["CLOUD_API_KEY"],
)

dispatcher = Dispatcher(
    router=router,
    simple_backend=local,
    complex_backend=cloud,
)

result = dispatcher.run(
    "What does HTTP 404 mean?"
)

print("Route:", result.route.label)
print("Backend:", result.backend_name)
print("Response:")
print(result.text)
