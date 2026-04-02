"""
Data Download Script for Earnings Call Intelligence System
===========================================================

Downloads public datasets for training the model.
Run this BEFORE training.

Datasets:
1. FinancialPhraseBank - Sentiment labels for financial sentences
2. FinQA - Financial question answering dataset
3. Custom Q&A pairs - Domain-specific examples

Usage:
    python data/download_data.py
"""

import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def setup_directories():
    """Create necessary directories."""
    RAW_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    logger.info(f"Directories created: {RAW_DIR}, {PROCESSED_DIR}")

def download_financial_phrasebank():
    """
    Download FinancialPhraseBank dataset.
    Source: https://huggingface.co/datasets/financial_phrasebank
    """
    logger.info("Downloading FinancialPhraseBank dataset...")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)
        
        logger.info(f"Loaded {len(dataset['train'])} examples from FinancialPhraseBank")
        
        train_data = []
        for item in dataset['train']:
            train_data.append({
                "text": item['sentence'],
                "label": item['label'],
                "source": "financial_phrasebank"
            })
        
        with open(RAW_DIR / "financial_phrasebank.json", "w") as f:
            json.dump(train_data, f, indent=2)
        
        logger.info(f"Saved to {RAW_DIR / 'financial_phrasebank.json'}")
        return len(train_data)
        
    except Exception as e:
        logger.error(f"Error downloading FinancialPhraseBank: {e}")
        return 0

def download_finqa():
    """
    Download FinQA dataset for financial Q&A.
    Source: https://huggingface.co/datasets/ParaLogic/finqa
    """
    logger.info("Downloading FinQA dataset...")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("ParaLogic/finqa", trust_remote_code=True)
        
        logger.info(f"Loaded {len(dataset['train'])} examples from FinQA")
        
        qa_data = []
        for item in dataset['train']:
            qa_data.append({
                "question": item.get('question', ''),
                "answer": item.get('answer', ''),
                "context": item.get('context', ''),
                "source": "finqa"
            })
        
        with open(RAW_DIR / "finqa.json", "w") as f:
            json.dump(qa_data, f, indent=2)
        
        logger.info(f"Saved to {RAW_DIR / 'finqa.json'}")
        return len(qa_data)
        
    except Exception as e:
        logger.error(f"Error downloading FinQA: {e}")
        return 0

def create_sample_earnings_data():
    """
    Create sample earnings call data with beat/miss labels.
    This includes realistic examples for training.
    """
    logger.info("Creating sample earnings call data...")
    
    sample_earnings = [
        {
            "company": "Apple",
            "quarter": "Q1 2024",
            "transcript": "Apple today reported fiscal Q1 2024 revenue of $119.6 billion, up 2% year over year. EPS of $2.18 compared to analyst estimates of $2.10. Services revenue reached a new all-time high of $22.3 billion, up 16% year over year. iPhone revenue was $69.1 billion, slightly below expectations. Mac revenue came in at $7.8 billion, up 0.6% year over year. For Q2, we expect revenue to be between $110 billion and $114 billion.",
            "eps_actual": 2.18,
            "eps_estimate": 2.10,
            "eps_beat": True,
            "revenue_actual": 119.6,
            "revenue_estimate": 117.5,
            "revenue_beat": True,
            "beat_miss": "BEAT"
        },
        {
            "company": "Tesla",
            "quarter": "Q3 2023",
            "transcript": "Tesla reported Q3 2023 revenue of $23.4 billion, up 9% year over year, surpassing analyst expectations of $23.2 billion. EPS of $0.66 exceeded estimates of $0.59. Automotive gross margin was 17.9%, down from 26% a year ago due to pricing pressures. We delivered 435,059 vehicles in the quarter, a new record. Energy storage deployments reached 4.0 GWh. Full Self-Driving beta now has over 400,000 users in North America.",
            "eps_actual": 0.66,
            "eps_estimate": 0.59,
            "eps_beat": True,
            "revenue_actual": 23.4,
            "revenue_estimate": 23.2,
            "revenue_beat": True,
            "beat_miss": "BEAT"
        },
        {
            "company": "Microsoft",
            "quarter": "Q2 2024",
            "transcript": "Microsoft Q2 FY2024 revenue was $62.0 billion, up 17% year over year. Net income was $21.9 billion, up 33% year over year. Azure and other cloud services grew 30%, ahead of expectations. LinkedIn revenue increased 9% to $4.0 billion. Windows revenue increased 13% due to Copilot integration. Gaming revenue was $7.1 billion, up 61% including Activision. We expect Q3 to grow 17-18% in revenue.",
            "eps_actual": 2.93,
            "eps_estimate": 2.78,
            "eps_beat": True,
            "revenue_actual": 62.0,
            "revenue_estimate": 61.5,
            "revenue_beat": True,
            "beat_miss": "BEAT"
        },
        {
            "company": "Amazon",
            "quarter": "Q4 2023",
            "transcript": "Amazon reported Q4 2023 net sales of $170.0 billion, up 13% year over year. AWS revenue was $25.0 billion, up 17% year over year, meeting expectations. Advertising revenue was $14.7 billion, up 27%. Operating income was $21.2 billion, up from $7.3 billion a year ago. EPS of $1.00 beat estimates of $0.78. For Q1 2024, we expect net sales between $138.0 billion and $143.0 billion.",
            "eps_actual": 1.00,
            "eps_estimate": 0.78,
            "eps_beat": True,
            "revenue_actual": 170.0,
            "revenue_estimate": 169.5,
            "revenue_beat": True,
            "beat_miss": "BEAT"
        },
        {
            "company": "Meta",
            "quarter": "Q4 2023",
            "transcript": "Meta Q4 2023 revenue was $40.1 billion, up 25% year over year, exceeding estimates. EPS of $5.33 beat estimates of $4.96. Family daily active people was 3.19 billion, up 8%. WhatsApp daily active users reached 100 million for the first time. Reality Labs lost $4.6 billion as expected. We expect Q1 2024 revenue between $34.0 billion and $37.0 billion. AI investments remain a priority with Llama gaining significant traction.",
            "eps_actual": 5.33,
            "eps_estimate": 4.96,
            "eps_beat": True,
            "revenue_actual": 40.1,
            "revenue_estimate": 39.0,
            "revenue_beat": True,
            "beat_miss": "BEAT"
        },
        {
            "company": "Netflix",
            "quarter": "Q3 2023",
            "transcript": "Netflix Q3 2023 revenue was $8.54 billion, up 7.8% year over year, slightly below expectations. EPS of $3.73 beat estimates of $3.49. Paid subscribers increased by 8.8 million, reaching 247 million. Ad-supported tier now has 23 million monthly active users globally. Operating margin was 22.4%, up from 17.9% a year ago. Free cash flow was $483 million. We expect Q4 revenue of $9.87 billion.",
            "eps_actual": 3.73,
            "eps_estimate": 3.49,
            "eps_beat": True,
            "revenue_actual": 8.54,
            "revenue_estimate": 8.60,
            "revenue_beat": False,
            "beat_miss": "BEAT"
        },
        {
            "company": "Google",
            "quarter": "Q4 2023",
            "transcript": "Alphabet Q4 2023 revenue was $86.3 billion, up 13% year over year, meeting expectations. Google Services revenue was $76.3 billion, up 14%. Google Cloud revenue was $9.2 billion, up 26%, beating estimates. EPS of $1.64 beat estimates of $1.59. YouTube advertising revenue was $9.2 billion, up 15%. Search revenue was $48.0 billion, up 12%. We returned $1.7 billion to shareholders in Q4.",
            "eps_actual": 1.64,
            "eps_estimate": 1.59,
            "eps_beat": True,
            "revenue_actual": 86.3,
            "revenue_estimate": 86.0,
            "revenue_beat": True,
            "beat_miss": "BEAT"
        },
        {
            "company": "Intel",
            "quarter": "Q4 2023",
            "transcript": "Intel Q4 2023 revenue was $15.4 billion, down 10% year over year, missing estimates of $15.2 billion. Data center revenue was $4.0 billion, down 7%. Client computing revenue was $8.8 billion, down 8%. EPS of $0.54 missed estimates of $0.64. Intel Foundry Services revenue was $291 million. We expect Q1 2024 revenue between $12.2 billion and $13.2 billion. Gaudi AI accelerators gaining traction with major customers.",
            "eps_actual": 0.54,
            "eps_estimate": 0.64,
            "eps_beat": False,
            "revenue_actual": 15.4,
            "revenue_estimate": 15.2,
            "revenue_beat": True,
            "beat_miss": "MISS"
        },
        {
            "company": "IBM",
            "quarter": "Q4 2023",
            "transcript": "IBM Q4 2023 revenue was $17.4 billion, up 4% year over year, meeting expectations. EPS of $3.92 beat estimates of $3.78. Software revenue was $7.6 billion, up 7%. Consulting revenue was $5.2 billion, up 6%. Infrastructure revenue was $4.6 billion, flat year over year. Free cash flow was $11.3 billion for the full year. Red Hat grew 17%. We expect mid-single-digit revenue growth for 2024.",
            "eps_actual": 3.92,
            "eps_estimate": 3.78,
            "eps_beat": True,
            "revenue_actual": 17.4,
            "revenue_estimate": 17.3,
            "revenue_beat": True,
            "beat_miss": "BEAT"
        },
        {
            "company": "JPMorgan",
            "quarter": "Q4 2023",
            "transcript": "JPMorgan Q4 2023 revenue was $39.9 billion, up 12% year over year, beating estimates. EPS of $3.97 beat estimates of $3.33. Investment banking fees were $2.3 billion, up 52%, beating expectations. Trading revenue was $5.8 billion, up 9%. Net interest income was $24.2 billion, up 34%. Provision for credit losses was $2.5 billion. The bank expects return on tangible common equity to be around 17% for 2024.",
            "eps_actual": 3.97,
            "eps_estimate": 3.33,
            "eps_beat": True,
            "revenue_actual": 39.9,
            "revenue_estimate": 39.0,
            "revenue_beat": True,
            "beat_miss": "BEAT"
        }
    ]
    
    with open(RAW_DIR / "sample_earnings.json", "w") as f:
        json.dump(sample_earnings, f, indent=2)
    
    logger.info(f"Created {len(sample_earnings)} sample earnings examples")
    return len(sample_earnings)

def generate_instruction_tuning_data():
    """
    Generate instruction tuning format data for training.
    Combines all raw data into training-ready format.
    """
    logger.info("Generating instruction tuning data...")
    
    all_data = []
    
    financial_phrasebank = []
    try:
        with open(RAW_DIR / "financial_phrasebank.json", "r") as f:
            financial_phrasebank = json.load(f)
    except FileNotFoundError:
        logger.warning("FinancialPhraseBank data not found, skipping...")
    
    sample_earnings = []
    try:
        with open(RAW_DIR / "sample_earnings.json", "r") as f:
            sample_earnings = json.load(f)
    except FileNotFoundError:
        logger.warning("Sample earnings data not found, skipping...")
    
    for item in financial_phrasebank:
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map.get(item["label"], "neutral")
        
        all_data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial analyst specializing in earnings call analysis. Analyze the provided text and determine the sentiment."
                },
                {
                    "role": "user",
                    "content": f"Analyze the sentiment of this financial statement: {item['text']}"
                },
                {
                    "role": "assistant",
                    "content": f"Sentiment: {sentiment.capitalize()}\n\nThis statement expresses a {sentiment} tone. " + 
                             ("Positive indicators include favorable financial metrics." if sentiment == "positive" else 
                              "Negative indicators include unfavorable financial metrics." if sentiment == "negative" else
                              "The statement is neutral and factual.")
                }
            ]
        })
    
    for item in sample_earnings:
        eps_status = "BEAT" if item["eps_beat"] else "MISS"
        revenue_status = "BEAT" if item["revenue_beat"] else "MISS"
        
        all_data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial analyst specializing in earnings call analysis. Analyze earnings calls and predict beat/miss outcomes."
                },
                {
                    "role": "user",
                    "content": f"Analyze this earnings call transcript for {item['company']} {item['quarter']}.\n\nTranscript:\n{item['transcript']}\n\nPredict whether they beat or missed analyst estimates for EPS and Revenue."
                },
                {
                    "role": "assistant",
                    "content": f"## Earnings Analysis: {item['company']} {item['quarter']}\n\n### Beat/Miss Prediction\n- **EPS:** {eps_status} (Actual: ${item['eps_actual']} vs Est: ${item['eps_estimate']})\n- **Revenue:** {revenue_status} (Actual: ${item['revenue_actual']}B vs Est: ${item['revenue_estimate']}B)\n\n### Overall: {item['beat_miss']}\n\n### Key Takeaways\nBased on the transcript analysis, {item['company']} {'exceeded' if item['beat_miss'] == 'BEAT' else 'fell short of'} expectations."
                }
            ]
        })
        
        all_data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial analyst. Answer questions about earnings calls based on the provided transcript."
                },
                {
                    "role": "user",
                    "content": f"Based on the {item['company']} {item['quarter']} earnings call, what were the key metrics mentioned?\n\nTranscript:\n{item['transcript']}"
                },
                {
                    "role": "assistant",
                    "content": f"## Key Metrics for {item['company']} {item['quarter']}\n\n### Earnings\n- EPS: ${item['eps_actual']} ({'Beat' if item['eps_beat'] else 'Miss'} vs ${item['eps_estimate']} estimate)\n\n### Revenue\n- Total: ${item['revenue_actual']}B ({'Beat' if item['revenue_beat'] else 'Miss'} vs ${item['revenue_estimate']}B estimate)\n\n### Summary\nOverall {item['beat_miss']} - {'Strong performance across segments.' if item['beat_miss'] == 'BEAT' else 'Mixed results with some challenges.'}"
                }
            ]
        })
    
    with open(PROCESSED_DIR / "train.jsonl", "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Generated {len(all_data)} training examples")
    return len(all_data)

def main():
    """Main download and processing pipeline."""
    logger.info("=" * 60)
    logger.info("Earnings Call Intelligence - Data Download")
    logger.info("=" * 60)
    
    setup_directories()
    
    logger.info("\n--- Step 1: Download FinancialPhraseBank ---")
    fp_count = download_financial_phrasebank()
    
    logger.info("\n--- Step 2: Download FinQA (optional) ---")
    finqa_count = download_finqa()
    
    logger.info("\n--- Step 3: Create Sample Earnings Data ---")
    earnings_count = create_sample_earnings_data()
    
    logger.info("\n--- Step 4: Generate Training Data ---")
    total_train = generate_instruction_tuning_data()
    
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"FinancialPhraseBank: {fp_count} examples")
    logger.info(f"FinQA: {finqa_count} examples")
    logger.info(f"Sample Earnings: {earnings_count} examples")
    logger.info(f"Total Training Examples: {total_train}")
    logger.info(f"\nData saved to: {PROCESSED_DIR / 'train.jsonl'}")
    logger.info("\nReady for training!")

if __name__ == "__main__":
    main()
