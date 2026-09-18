# Python Rerouting Library

A lightweight Python library that routes user queries between a local Llama model and a cloud LLM using privacy rules and semantic complexity classification.

## Routing Architecture

Version 0.2 introduces a privacy-first routing policy.

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

