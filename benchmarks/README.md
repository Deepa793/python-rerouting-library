# Benchmarks

Keep evaluation datasets and evaluator scripts here.

Recommended next benchmark:
- 150 queries
- 75 simple
- 75 complex
- include difficult boundary cases
- use group/family-aware cross-validation to reduce leakage

Current recorded baseline from the 50-query suite:

TF-IDF + Logistic Regression:
- Accuracy: 0.9000
- Precision (complex): 0.8333
- Recall (complex): 1.0000
- F1 (complex): 0.9091
- CPU median: ~0.531 ms/query
- Wall median: ~0.500 ms/query

MiniLM + Logistic Regression:
- Accuracy: 1.0000
- Precision (complex): 1.0000
- Recall (complex): 1.0000
- F1 (complex): 1.0000
- CPU median: ~149.609 ms aggregate process CPU
- Wall median: ~6.416 ms/query

Do not treat 100% on the 50-query suite as general routing accuracy.
