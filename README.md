# Python Rerouting Library

A lightweight Python library that routes user queries between a local Llama model and a cloud LLM using privacy rules and semantic complexity classification.

## What Changed in v0.2.0

Version 0.2 adds a **privacy-first routing layer** in front of the semantic complexity router.

The key rule is:

> Privacy policy takes precedence over complexity routing.

If a query contains detected sensitive information, the query is routed directly to the local model and is never sent to the cloud.

---

## Routing Architecture

```text
Query
  ↓
Privacy Detector
  ↓
Sensitive data detected?
  │
  ├── YES
  │     ↓
  │   privacy_override
  │     ↓
  │   Local Llama only
  │     ↓
  │   if local fails → DispatchError
  │
  │   NEVER cloud
  │
  └── NO
        ↓
      MiniLM Semantic Router
        ↓
      Logistic Regression
        ↓
      simple / uncertain / complex
```

For clean queries, normal routing continues:

```text
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

---

## Privacy-First Routing

Before complexity scoring, every query is checked by `PrivacyDetector`.

The current detector looks for common patterns including:

* Email addresses
* US-style phone numbers
* Social Security numbers
* Common API-key formats
* Credit-card numbers

Credit-card candidates are additionally validated using the **Luhn checksum** to reduce false positives.

If sensitive information is detected:

```text
Query
  ↓
PrivacyDetector
  ↓
privacy_override
  ↓
Local Llama
```

The MiniLM semantic router is not called.

The cloud backend is not called.

If the local backend fails:

```text
privacy_override
  ↓
Local backend failure
  ↓
DispatchError
```

There is intentionally **no cloud fallback** for privacy-sensitive queries.

The detector reports only privacy categories such as:

```text
email
phone
ssn
api_key
credit_card
```

It does not return the matched sensitive value.

> The privacy detector is a lightweight policy layer based on known patterns. It is not intended to replace a full Data Loss Prevention, compliance, or enterprise sensitive-data classification system.

---

## Semantic Router

Queries that pass the privacy check continue to the semantic router.

The router uses:

* `sentence-transformers/all-MiniLM-L6-v2`
* 384-dimensional semantic embeddings
* Logistic Regression
* Configurable uncertainty thresholds

The query is converted into an embedding and passed to the trained classifier.

The classifier produces:

```text
P(complex)
```

This probability is then used by the routing policy.

---

## Default Complexity Routing Policy

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

---

## Routing Priority

The system now contains two routing layers:

```text
1. Privacy policy
       ↓
2. Complexity policy
```

Privacy always wins.

For example, a query such as:

```text
Design a multi-region database architecture
for alice@example.com with automatic failover.
```

looks like a complex query.

In v0.1 it would likely be routed to the cloud.

In v0.2:

```text
email detected
    ↓
privacy_override
    ↓
local only
```

The semantic complexity classifier is not called.

---

## Failure Policy

### Privacy-sensitive query

```text
Local Llama
    ↓ failure
DispatchError
```

Cloud fallback is prohibited.

### Clean simple query

```text
Local Llama
    ↓ failure
Cloud fallback
```

### Clean uncertain query

```text
Cloud
    ↓ failure
DispatchError
```

### Clean complex query

```text
Cloud
    ↓ failure
DispatchError
```

Complex and uncertain queries are not silently downgraded to the local model.

---

## Project Structure

```text
python-rerouting-library/
├── src/
│   └── python_rerouting_library/
│       ├── __init__.py
│       ├── router.py
│       ├── privacy.py
│       ├── dispatcher.py
│       ├── training.py
│       ├── config.py
│       ├── exceptions.py
│       └── backends/
│           ├── __init__.py
│           ├── local_llama.py
│           └── cloud.py
├── tests/
│   ├── test_dispatcher.py
│   ├── test_privacy.py
│   ├── test_router_decision.py
│   └── test_router_thresholds.py
├── examples/
├── benchmarks/
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

The generated router classifier is stored under:

```text
artifacts/
```

and is intentionally excluded from Git.

---

## Requirements

* Python 3.10+
* Router classifier artifact
* Local GGUF model for local Llama inference
* OpenAI-compatible cloud API for cloud routing

---

## Installation

From the project folder:

```powershell
python -m pip install -e ".[dev]"
```

If using the local Llama backend:

```powershell
python -m pip install -e ".[local-llama]"
```

`llama-cpp-python` is an optional dependency.

Cloud-only users do not need to install it.

---

## Configuration

The library reads runtime settings from environment variables.

Example:

```powershell
$env:LLAMA_MODEL_PATH="C:\models\llama3.2\model.gguf"

$env:ROUTER_CLASSIFIER_PATH="artifacts\router_classifier.joblib"

$env:CLOUD_BASE_URL="https://api.openai.com/v1"
$env:CLOUD_MODEL="gpt-5.4-nano"
$env:CLOUD_API_KEY="YOUR_API_KEY"

$env:LOCAL_MAX_TOKENS="128"
$env:CLOUD_MAX_TOKENS="256"

$env:ROUTER_SIMPLE_THRESHOLD="0.40"
$env:ROUTER_COMPLEX_THRESHOLD="0.60"
```

See `.env.example` for the available settings.

Never commit real API keys or credentials.

---

## Classifier Artifact

The package does not ship with a default production complexity classifier.

Complexity classification is workload-dependent. What should be considered a simple or complex query can vary depending on:

- the local model being used
- the cloud model being used
- application domain
- latency requirements
- cost policy
- desired routing behavior

Users should therefore train or provide a compatible classifier artifact.

When using `Router` directly:

```python
from python_rerouting_library import Router

router = Router(
    classifier_path="path/to/router_classifier.joblib"
)

## Train the Router

The current example training dataset is:

```text
benchmarks/router_queries_50.csv
```

Train and save the classifier with:

```powershell
python -m python_rerouting_library.training `
    --csv benchmarks\router_queries_50.csv `
    --output artifacts\router_classifier.joblib
```

The training pipeline uses MiniLM embeddings and Logistic Regression.

The router thresholds are runtime routing policy and are not stored as a training-time decision threshold.

---

## Run the Full Example

After configuring the environment and training the router:

```powershell
python .\examples\test_full_dispatcher.py
```

A clean simple query should normally route to:

```text
Backend: local-llama
```

A clean complex query should normally route to:

```text
Backend: cloud-api
```

A detected privacy-sensitive query should route to:

```text
Route: privacy_override
Backend: local-llama
```

---

## Run Tests

```powershell
python -m pytest -v
```

Current v0.2 regression suite:

```text
26 passed
```

The tests cover:

* simple → local
* complex → cloud
* uncertain → cloud
* clean simple local failure → cloud fallback
* cloud failure handling
* uncertainty threshold boundaries
* structured route decisions
* email detection
* phone-number detection
* SSN detection
* API-key detection
* credit-card detection
* Luhn validation
* multiple privacy categories
* privacy override before semantic routing
* verification that privacy queries do not call the router
* verification that privacy queries do not fall back to cloud

The automated unit tests do not call the OpenAI API.

---

## Route Decision Behavior

Normal semantic routes contain complexity information:

```text
label
confidence
complex_probability
latency_ms
```

For a privacy override:

```text
label = privacy_override
confidence = None
complex_probability = None
```

This is intentional.

The complexity classifier never ran, so the library does not invent a complexity probability.

---

## Privacy Override Example

Conceptually:

```python
result = dispatcher.run(
    "Please review account details for alice@example.com"
)

print(result.route.label)
print(result.backend_name)
print(result.privacy_categories)
```

Expected behavior:

```text
privacy_override
local-llama
('email',)
```

The query is not sent to the cloud.

---

## Version History

### v0.2.0

Adds:

* Privacy-first routing
* `PrivacyDetector`
* `PrivacyDecision`
* `privacy_override`
* Local-only processing for detected sensitive data
* No cloud fallback for privacy-sensitive queries
* Email detection
* Phone-number detection
* SSN detection
* API-key pattern detection
* Credit-card detection with Luhn validation
* Privacy-category observability
* Expanded regression tests

### v0.1.0

Introduced:

* MiniLM semantic query embeddings
* Logistic Regression complexity classifier
* Simple / uncertain / complex routing
* Local Llama backend
* Cloud LLM backend
* Local-to-cloud fallback for simple queries
* Custom backend exceptions
* Centralized configuration
* Automated regression tests

---

## Design Principle

The library separates three concerns:

```text
Policy
  ↓
Routing
  ↓
Execution
```

In v0.2:

```text
Privacy Detector
      ↓
Semantic Router
      ↓
Dispatcher
      ↓
Local / Cloud Backend
```

This means model-selection optimization happens only after the privacy policy allows the query to continue through the normal routing pipeline.

---

## Current Limitations

The current privacy detector is intentionally lightweight.

It does not yet provide:

* Named-entity recognition
* Address detection
* Medical-record detection
* International identity-number detection
* Context-aware secret detection
* Configurable privacy policies
* User-defined sensitive-data patterns
* Enterprise DLP integration

These are possible future extensions.

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
