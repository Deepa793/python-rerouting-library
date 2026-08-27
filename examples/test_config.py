from python_rerouting_library.config import Settings


settings = Settings.from_env()

print("Router classifier:", settings.router_classifier_path)
print("Llama model:", settings.llama_model_path)

print("Cloud base URL:", settings.cloud_base_url)
print("Cloud model:", settings.cloud_model)

print("Cloud API key configured:", bool(settings.cloud_api_key))

print("Local max tokens:", settings.local_max_tokens)
print("Cloud max tokens:", settings.cloud_max_tokens)

print("Router simple threshold:", settings.router_simple_threshold)
print("Router complex threshold:", settings.router_complex_threshold)