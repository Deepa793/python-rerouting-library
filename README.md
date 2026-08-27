# Python Rerouting Library

A lightweight Python library that routes user queries to either a local Llama model or a cloud LLM based on query complexity.

## Routing Architecture

```text
Query
  ↓
MiniLM Semantic Router
  ↓
simple / uncertain / complex

simple
  ↓
Local Llama
  ↓
if local fails → Cloud fallback

uncertain
  ↓
Cloud

complex
  ↓
Cloud
```

The semantic router uses:

* `sentence-transformers/all-MiniLM-L6-v2`
* Logistic Regression
* Configurable uncertainty thresholds

## Default Routing Policy

```text
P(complex) < 0.40
    → simple
    → local Llama

0.40 ≤ P(complex) ≤ 0.60
    → uncertain
    → cloud

P(complex) > 0.60
    → complex
    → cloud
```

The thresholds are configurable.

## Failure Policy

For simple queries:

```text
Local Llama
    ↓ failure
Cloud fallback
```

For complex or uncertain queries:

```text
Cloud
    ↓ failure
DispatchError
```

Complex queries are not silently downgraded to the local model.

## Project Structure

```text
python_rerouting_library_v0_1/
├── src/
│   └── python_rerouting_library/
│       ├── router.py
│       ├── dispatcher.py
│       ├── training.py
│       ├── config.py
│       ├── exceptions.py
│       └── backends/
│           ├── local_llama.py
│           └── cloud.py
├── tests/
├── examples/
├── benchmarks/
├── artifacts/
├── pyproject.toml
├── .env.example
└── README.md
```

## Requirements

* Python 3.12+
* Local GGUF model for the local backend
* OpenAI-compatible cloud API
* Router classifier artifact

## Installation

From the project folder:

```powershell
python -m pip install -e ".[dev]"
```

If using the local Llama backend:

```powershell
python -m pip install -e ".[local-llama]"
```

## Configuration

The library reads configuration from environment variables.

Example:

```powershell
$env:LLAMA_MODEL_PATH="C:\models\llama3.2\model.gguf"
$env:CLOUD_API_KEY="YOUR_API_KEY"
$env:CLOUD_BASE_URL="https://api.openai.com/v1"
$env:CLOUD_MODEL="gpt-5.4-nano"
$env:LOCAL_MAX_TOKENS="128"
$env:CLOUD_MAX_TOKENS="256"
$env:ROUTER_SIMPLE_THRESHOLD="0.40"
$env:ROUTER_COMPLEX_THRESHOLD="0.60"
```

See `.env.example` for the full configuration.

Do not commit real API keys.

## Train the Router

The current training dataset is:

```text
benchmarks/router_queries_50.csv
```

Train and save the classifier with:

```powershell
python -m python_rerouting_library.training `
    --csv benchmarks\router_queries_50.csv `
    --output artifacts\router_classifier.joblib
```

The generated classifier artifact is intentionally excluded from Git.

## Run the Full Example

After configuring the environment and training the classifier:

```powershell
python .\examples\test_full_dispatcher.py
```

A simple query should route to:

```text
Backend: local-llama
```

A complex query should route to:

```text
Backend: cloud-api
```

## Run Tests

```powershell
python -m pytest -v
```

Current v0.1 regression suite:

```text
13 passed
```

The tests cover:

* simple → local
* complex → cloud
* uncertain → cloud
* local failure → cloud fallback
* cloud failure handling
* uncertainty boundary behavior
* structured route decisions

The automated unit tests do not call the OpenAI API.

## Current Status

v0.1 provides:

* Semantic query routing
* Local Llama inference
* Cloud LLM routing
* Uncertainty-aware dispatch
* Local-to-cloud fallback
* Custom backend exceptions
* Centralized configuration
* Automated regression tests


