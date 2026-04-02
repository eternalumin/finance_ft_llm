# Results Directory

This directory contains evaluation results and benchmark comparisons.

## Files

- `evaluation_results_*.json` - Individual evaluation run results
- `BENCHMARK_REPORT.md` - Comparison with baseline models

## Running Evaluation

```bash
# Evaluate the model
python evaluation/evaluate.py

# Compare with baselines
python evaluation/benchmark.py
```

## Output Format

### Evaluation Results (`evaluation_results_*.json`)

```json
{
  "beat_miss": {
    "accuracy": 0.87,
    "f1_score": 0.86,
    "precision": 0.85,
    "recall": 0.87,
    "num_samples": 50,
    "avg_confidence": 0.82
  },
  "response_quality": {
    "avg_cosine_similarity": 0.82,
    "num_samples": 20
  },
  "summary": {
    "beat_miss_accuracy": 0.87,
    "f1_score": 0.86,
    "response_quality": 0.82
  },
  "metadata": {
    "model_path": "training/outputs/checkpoint-2500",
    "timestamp": "2024-01-15_10:30:00",
    "device": "NVIDIA T4"
  }
}
```

### Benchmark Report (`BENCHMARK_REPORT.md`)

Markdown table comparing our model against:
- Base Llama-3.2-3B-Instruct
- BloombergGPT
- FinGPT

## Metrics Explained

| Metric | Description | Target |
|--------|-------------|--------|
| Beat/Miss Accuracy | % correct predictions | >85% |
| F1 Score | Harmonic mean of precision/recall | >0.80 |
| Response Quality | Cosine similarity vs reference | >0.75 |
| Avg Confidence | Average model confidence | 0.7-0.9 |
