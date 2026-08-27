import os

from python_rerouting_library.backends import CloudBackend


cloud = CloudBackend(
    base_url="https://api.openai.com/v1",
    model="gpt-5.4-nano",
    api_key=os.environ["CLOUD_API_KEY"],
    max_tokens=128,
    temperature=0.0,
)

query = "Explain why a multi-region database architecture needs a failover strategy."

print("Query:", query)

response = cloud.generate(query)

print("Response:")
print(response)
print("Type:", type(response))