from python_rerouting_library.backends import LocalLlamaBackend

local = LocalLlamaBackend(
    model_path=r"C:\models\llama3.2\model.gguf",
    temperature=0.0,
    max_tokens=128,
)

query = "What does HTTP 404 mean?"

print("Query:", query)

response = local.generate(query)

print("Response:")
print(response)
print("Type:", type(response))