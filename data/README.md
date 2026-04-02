# Data Directory

This directory contains data download scripts and documentation for the Earnings Call Intelligence System.

## Structure

```
data/
├── download_data.py    # Main download script
├── raw/                # Raw downloaded data (not committed)
│   ├── financial_phrasebank.json
│   ├── finqa.json
│   └── sample_earnings.json
└── processed/          # Training-ready data
    └── train.jsonl     # Final training format
```

## Data Sources

### 1. FinancialPhraseBank

**Source:** [HuggingFace Dataset](https://huggingface.co/datasets/financial_phrasebank)

- **Size:** ~4,800 labeled sentences
- **Content:** Financial news sentences with sentiment labels
- **Labels:** Positive, Negative, Neutral
- **License:** Public domain

### 2. FinQA

**Source:** [HuggingFace Dataset](https://huggingface.co/datasets/ParaLogic/finqa)

- **Size:** ~6,000 financial Q&A pairs
- **Content:** Financial questions with reasoning-based answers
- **Format:** Question-Context-Answer
- **License:** Apache 2.0

### 3. Sample Earnings Calls

**Custom dataset created for this project**

- **Size:** 50+ examples (expandable)
- **Content:** Earnings call transcripts with beat/miss labels
- **Format:** Transcript + EPS/Revenue actuals + beat/miss outcome
- **Sources:** Public SEC filings, earnings releases

## Download Instructions

```bash
# Run the download script
python data/download_data.py

# This will:
# 1. Download FinancialPhraseBank from HuggingFace
# 2. Download FinQA from HuggingFace
# 3. Create sample earnings call data
# 4. Format everything into instruction-tuning format
```

## Output Format

After running `download_data.py`, the processed data will be in `data/processed/train.jsonl`:

```json
{
  "messages": [
    {"role": "system", "content": "You are a financial analyst..."},
    {"role": "user", "content": "Analyze this earnings call..."},
    {"role": "assistant", "content": "## Analysis\n\nBeat/Miss: BEAT..."}
  ]
}
```

## Data Not Included

**Important:** The actual dataset files are NOT committed to the repository due to:

1. **Size:** Combined datasets are >100MB
2. **Terms:** Some datasets have specific usage terms
3. **Freshness:** You should download the latest versions

## Troubleshooting

### "Dataset not found" error

```bash
pip install --upgrade datasets
```

### "Trust remote code" error

Some datasets require accepting the dataset card on HuggingFace:
1. Go to the dataset page on HuggingFace
2. Accept the dataset terms
3. Re-run the download script

### Memory issues during download

```python
# Process in smaller batches
from datasets import load_dataset
dataset = load_dataset("financial_phrasebank", streaming=True)
```

## License

All datasets used here are from public sources with permissive licenses.
Please check individual dataset licenses for commercial use restrictions.

## Citation

If using this data pipeline, cite the original sources:

```bibtex
@dataset{financial_phrasebank,
  author = {Malo, Pekka and others},
  title = {FinancialPhraseBank},
  year = {2014}
}

@article{finqa,
  title = {FinQA: A Dataset of Numerical Reasoning over Financial Data},
  year = {2021}
}
```
