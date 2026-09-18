# Python Rerouting Library

A lightweight Python library for routing user queries between local and cloud LLM backends using **privacy-aware policy checks** and **semantic complexity classification**.

Current version: **0.2.1**

PyPI:

```text
https://pypi.org/project/python-rerouting-library/
```

GitHub:

```text
https://github.com/Deepa793/python-rerouting-library
```

---

## Overview

Python Rerouting Library provides two routing layers:

```text
Query
  ↓
Privacy Policy
  ↓
Semantic Complexity Routing
  ↓
Local or Cloud Backend
```

The first layer checks for supported sensitive-data patterns.

The second layer determines whether a clean query is simple, uncertain, or complex.

The key design principle is:

> A detected privacy match takes precedence over complexity routing.

---

# Routing Architecture

```text
Query
  ↓
PrivacyDetector
  ↓
Supported sensitive pattern detected?
  │
  ├── YES
  │     ↓
  │   privacy_override
  │     ↓
  │   Local backend only
  │     ↓
  │   local failure → DispatchError
  │
  │   no cloud fallback for this matched query
  │
  └── NO
        ↓
      MiniLM Embedding
        ↓
      Logistic Regression
        ↓
      P(complex)
        ↓
      simple / uncertain / complex
```

Normal complexity routing:

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

# Privacy-First Routing

Before complexity scoring, each query is evaluated by `PrivacyDetector`.

The current detector looks for supported patterns including:

* Email addresses
* US-style phone numbers
* Social Security numbers
* Common API-key formats
* Credit-card numbers

Credit-card candidates are additionally validated using the **Luhn checksum** to reduce false positives.

If `PrivacyDetector` identifies one or more supported sensitive-data patterns, the dispatcher assigns:

```text
privacy_override
```

For a query that triggers `privacy_override`:

* the MiniLM complexity router is not invoked;
* the configured local backend is used;
* the dispatcher does not invoke the cloud backend;
* cloud fallback is disabled for that matched request;
* if local processing fails, a `DispatchError` is raised.

Example:

```text
Design a multi-region database architecture
for alice@example.com with automatic failover.
```

The query may look complex, but the email pattern is detected first:

```text
email detected
    ↓
privacy_override
    ↓
local backend
```

The complexity classifier is not called for that request.

---

## Privacy Categories

`PrivacyDetector` currently reports categories such as:

```text
email
phone
ssn
api_key
credit_card
```

It does not intentionally return the matched sensitive value through `PrivacyDecision`.

Example:

```python
from python_rerouting_library import PrivacyDetector

detector = PrivacyDetector()

decision = detector.detect(
    "Contact alice@example.com"
)

print(decision)
```

Expected:

```text
PrivacyDecision(
    is_sensitive=True,
    categories=('email',)
)
```

---

# Important Privacy Limitations

`PrivacyDetector` is a lightweight, pattern-based safeguard.

It is **not a complete privacy, security, DLP, or compliance system**.

Detection is heuristic and may produce both:

* false positives
* false negatives

Sensitive information that does not match one of the supported patterns may continue through normal routing and, depending on the resulting routing decision, may be sent to a configured cloud backend.

The current detector does not comprehensively identify:

* names or named entities
* street addresses
* medical or health information
* international identity numbers
* arbitrary credentials or secrets
* context-dependent sensitive information
* obfuscated sensitive information
* unusual representations of otherwise supported patterns

Applications handling regulated, confidential, or highly sensitive information should use additional controls appropriate to their environment, such as:

* enterprise DLP
* data classification
* redaction
* access controls
* auditing
* provider-specific privacy controls
* application-level validation

The privacy layer should therefore be treated as an **additional routing safeguard**, not as a guarantee that all sensitive information will be detected or prevented from leaving the local environment.

---

# Semantic Router

Queries that do not trigger a privacy override continue to the semantic router.

The router uses:

* `sentence-transformers/all-MiniLM-L6-v2`
* 384-dimensional semantic embeddings
* Logistic Regression
* configurable uncertainty thresholds

The query is converted into an embedding.

The trained classifier then estimates:

```text
P(complex)
```

This probability is used by the routing policy.

---

# Default Complexity Routing Policy

```text
P(complex) < 0.40
    → simple
    → local backend

0.40 ≤ P(complex) ≤ 0.60
    → uncertain
    → cloud backend

P(complex) > 0.60
    → complex
    → cloud backend
```

The thresholds are configurable.

---

# Routing Priority

The routing policy contains two layers:

```text
1. Privacy policy
       ↓
2. Complexity policy
```

A detected privacy match takes precedence over complexity routing.

This means model-selection optimization occurs only after the privacy detector allows the request to continue through the normal semantic-routing pipeline.

---

# Failure Policy

## Detected privacy-sensitive query

```text
privacy_override
    ↓
Local backend
    ↓ failure
DispatchError
```

No cloud fallback is attempted for that matched request.

---

## Clean simple query

```text
simple
    ↓
Local backend
    ↓ failure
Cloud fallback
```

---

## Clean uncertain query

```text
uncertain
    ↓
Cloud backend
    ↓ failure
DispatchError
```

---

## Clean complex query

```text
complex
    ↓
Cloud backend
    ↓ failure
DispatchError
```

Complex and uncertain queries are not silently downgraded to the local model.

---

# Route Decision Behavior

Normal semantic routes contain:

```text
label
confidence
complex_probability
latency_ms
```

Example:

```text
label = simple
confidence = 0.68
complex_probability = 0.32
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

# Project Structure

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
│
├── tests/
│   ├── test_classifier_artifact.py
│   ├── test_dispatcher.py
│   ├── test_privacy.py
│   ├── test_router_decision.py
│   └── test_router_thresholds.py
│
├── examples/
├── benchmarks/
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

Generated classifier artifacts are normally stored under:

```text
artifacts/
```

and are intentionally excluded from Git.

---

# Requirements

* Python 3.10+
* A compatible router classifier artifact
* A local GGUF model if using the local Llama backend
* A compatible cloud API if using cloud routing

---

# Installation

Install from PyPI:

```powershell
pip install python-rerouting-library
```

Install a specific version:

```powershell
pip install python-rerouting-library==0.2.1
```

If using the local Llama backend:

```powershell
pip install "python-rerouting-library[local-llama]"
```

For development:

```powershell
python -m pip install -e ".[dev]"
```

`llama-cpp-python` is an optional dependency.

Cloud-only users do not need to install it.

---

# Configuration

The library can read runtime configuration from environment variables.

Example:

```powershell
$env:LLAMA_MODEL_PATH="C:\models\llama3.2\model.gguf"

$env:ROUTER_CLASSIFIER_PATH="C:\path\to\router_classifier.joblib"

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

# Classifier Artifact

The package does **not** ship with a default production complexity classifier.

Complexity classification is workload-dependent.

What should be considered simple or complex can vary based on:

* local model capability
* cloud model capability
* application domain
* latency requirements
* cost policy
* desired routing behavior

Users should therefore train or provide a compatible classifier artifact.

When using `Router` directly:

```python
from python_rerouting_library import Router

router = Router(
    classifier_path="path/to/router_classifier.joblib"
)
```

When using environment-based configuration:

```powershell
$env:ROUTER_CLASSIFIER_PATH="C:\path\to\router_classifier.joblib"
```

If the classifier does not exist, the library raises an actionable error explaining how to provide or train one.

---

# Train the Router

The repository includes an example development dataset:

```text
benchmarks/router_queries_50.csv
```

Train and save a classifier with:

```powershell
python -m python_rerouting_library.training `
    --csv benchmarks\router_queries_50.csv `
    --output artifacts\router_classifier.joblib
```

The training pipeline uses:

```text
Query
  ↓
MiniLM embedding
  ↓
Logistic Regression
  ↓
Classifier artifact
```

The included benchmark dataset is intended as an **example and development baseline**, not as a universal production routing model.

The routing thresholds are runtime policy and are not stored as a training-time decision threshold.

---

# Lazy Loading

The Sentence Transformers stack is loaded only when semantic routing actually requires it.

This means lightweight functionality such as:

```python
from python_rerouting_library import PrivacyDetector
```

can be used without immediately initializing the embedding-model stack.

Similarly, cloud-backend imports do not require `llama-cpp-python`.

This keeps optional functionality isolated and improves package usability across different environments.

---

# Run the Full Example

After configuring the environment and training or providing the router classifier:

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

A query that matches a supported privacy pattern should produce behavior similar to:

```text
Route: privacy_override
Backend: local-llama
```

---

# Run Tests

```powershell
python -m pytest -v
```

Current v0.2.1 regression suite:

```text
29 passed
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
* verification that matched privacy queries do not invoke the semantic router
* verification that matched privacy queries do not fall back to cloud
* explicit classifier-artifact configuration
* missing-classifier error handling

The automated unit tests do not call the OpenAI API.

---

# Privacy Override Example

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

Because the email pattern is detected, this request takes the `privacy_override` path and the dispatcher does not invoke the cloud backend.

---

# Version History

## v0.2.1

Adds PyPI release hardening and clearer package behavior:

* Published distribution on PyPI
* Explicit classifier-artifact requirement
* Improved missing-classifier errors
* Lazy loading of the Sentence Transformers stack
* Fresh-environment wheel validation
* TestPyPI validation
* Improved privacy documentation
* Expanded regression suite to 29 tests

PyPI:

```text
https://pypi.org/project/python-rerouting-library/0.2.1/
```

---

## v0.2.0

Introduced privacy-first routing:

* `PrivacyDetector`
* `PrivacyDecision`
* `privacy_override`
* Local-only processing for matched privacy patterns
* No cloud fallback for matched privacy-override requests
* Email detection
* Phone-number detection
* SSN detection
* API-key pattern detection
* Credit-card detection with Luhn validation
* Privacy-category observability

---

## v0.1.0

Introduced:

* MiniLM semantic query embeddings
* Logistic Regression complexity classifier
* simple / uncertain / complex routing
* Local Llama backend
* Cloud LLM backend
* Local-to-cloud fallback for simple queries
* Custom backend exceptions
* Centralized configuration
* Automated regression tests

---

# Design Principle

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
PrivacyDetector
      ↓
Semantic Router
      ↓
Dispatcher
      ↓
Local / Cloud Backend
```

The privacy detector acts as a policy gate.

The semantic router handles model-selection optimization.

The dispatcher handles backend execution and fallback behavior.

---

# Current Limitations

The project is intentionally lightweight and still evolving.

Current limitations include:

* privacy detection is regex/pattern based
* privacy detection may miss sensitive information
* no named-entity recognition
* no address detection
* no medical-record classification
* no international identity-number coverage
* no context-aware privacy classification
* no configurable privacy-rule registry yet
* no enterprise DLP integration
* no bundled production complexity classifier

These are potential areas for future development.

---

# Security and Privacy Guidance

Do not treat this library as the sole privacy or security boundary for applications handling confidential, regulated, or highly sensitive information.

Before production deployment, evaluate:

* your privacy requirements
* your data-classification requirements
* your cloud-provider policies
* your logging behavior
* local-model security
* secret management
* access controls
* regulatory or contractual requirements

Applications should apply defense-in-depth rather than relying solely on the routing layer.

---

# License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
