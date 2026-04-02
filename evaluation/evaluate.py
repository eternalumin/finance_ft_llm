"""
Evaluation Script for Earnings Call Intelligence System
=======================================================

Evaluates the fine-tuned model on various metrics.

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --model_path training/outputs/checkpoint-2500
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "training/outputs/earnings-intelligence-v1"
RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_model_and_tokenizer(model_path: str):
    """Load fine-tuned model and tokenizer."""
    logger.info(f"Loading model from {model_path}")
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        logger.info("Please train the model first: python training/train.py")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model.eval()
    
    return model, tokenizer

def create_pipeline(model, tokenizer):
    """Create inference pipeline."""
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )

def parse_beat_miss_prediction(response: str) -> Tuple[str, float]:
    """
    Parse beat/miss prediction from model response.
    
    Returns:
        Tuple of (prediction, confidence)
    """
    response_upper = response.upper()
    
    if "BEAT" in response_upper and "MISS" not in response_upper:
        prediction = "BEAT"
    elif "MISS" in response_upper and "BEAT" not in response_upper:
        prediction = "MISS"
    elif "MEET" in response_upper:
        prediction = "MEET"
    else:
        prediction = "UNKNOWN"
    
    confidence = 0.5
    if "CONFIDENCE" in response_upper:
        for word in response.split():
            if word.replace("%", "").isdigit():
                confidence = int(word.replace("%", "")) / 100
                break
    
    return prediction, confidence

def evaluate_beat_miss(
    pipeline,
    tokenizer,
    test_data: List[Dict],
    max_samples: int = 50
) -> Dict[str, Any]:
    """Evaluate beat/miss prediction accuracy."""
    logger.info("Evaluating Beat/Miss Prediction...")
    
    predictions = []
    actuals = []
    confidences = []
    
    for i, item in enumerate(test_data[:max_samples]):
        if "messages" not in item:
            continue
        
        messages = item["messages"]
        if len(messages) < 3:
            continue
        
        transcript = messages[1].get("content", "")
        
        prompt = f"""Analyze this earnings call transcript and predict whether the company BEAT or MISSED analyst estimates.

Transcript:
{transcript[:1500]}

Respond with BEAT or MISS and your confidence level (0-100%)."""
        
        try:
            response = pipeline(prompt, return_full_text=False)[0]["generated_text"]
            
            pred, conf = parse_beat_miss_prediction(response)
            predictions.append(pred)
            confidences.append(conf)
            
            if "raw" in item:
                actual = item["raw"].get("beat_miss", "UNKNOWN")
            else:
                actual = "UNKNOWN"
            actuals.append(actual)
            
        except Exception as e:
            logger.warning(f"Error on sample {i}: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{min(len(test_data), max_samples)} samples")
    
    valid_indices = [i for i, p in enumerate(predictions) if p != "UNKNOWN"]
    
    if len(valid_indices) < 5:
        logger.warning("Not enough valid predictions for evaluation")
        return {"accuracy": 0.0, "f1": 0.0}
    
    filtered_preds = [predictions[i] for i in valid_indices]
    filtered_actuals = [actuals[i] for i in valid_indices]
    
    accuracy = accuracy_score(filtered_actuals, filtered_preds)
    f1 = f1_score(filtered_actuals, filtered_preds, average="weighted", labels=["BEAT", "MISS", "MEET"])
    precision = precision_score(filtered_actuals, filtered_preds, average="weighted", labels=["BEAT", "MISS", "MEET"], zero_division=0)
    recall = recall_score(filtered_actuals, filtered_preds, average="weighted", labels=["BEAT", "MISS", "MEET"], zero_division=0)
    
    results = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "num_samples": len(filtered_preds),
        "avg_confidence": np.mean(confidences),
        "classification_report": classification_report(filtered_actuals, filtered_preds, labels=["BEAT", "MISS", "MEET"])
    }
    
    logger.info(f"\nBeat/Miss Prediction Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  Avg Confidence: {np.mean(confidences):.2f}")
    
    return results

def evaluate_response_quality(
    pipeline,
    tokenizer,
    test_data: List[Dict],
    max_samples: int = 20
) -> Dict[str, float]:
    """Evaluate response quality using cosine similarity."""
    logger.info("Evaluating Response Quality...")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = []
    
    for i, item in enumerate(test_data[:max_samples]):
        if "messages" not in item or len(item["messages"]) < 3:
            continue
        
        messages = item["messages"]
        
        if len(messages) >= 3:
            reference = messages[2].get("content", "")
            
            prompt = messages[1].get("content", "")
            
            try:
                response = pipeline(prompt, return_full_text=False)[0]["generated_text"]
                
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform([response, reference])
                similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
                similarities.append(similarity)
                
            except Exception as e:
                logger.warning(f"Error on sample {i}: {e}")
                continue
        
        if (i + 1) % 5 == 0:
            logger.info(f"Processed {i + 1}/{min(len(test_data), max_samples)} samples")
    
    avg_similarity = np.mean(similarities) if similarities else 0.0
    
    results = {
        "avg_cosine_similarity": avg_similarity,
        "num_samples": len(similarities),
        "min_similarity": np.min(similarities) if similarities else 0,
        "max_similarity": np.max(similarities) if similarities else 0,
    }
    
    logger.info(f"\nResponse Quality Results:")
    logger.info(f"  Avg Cosine Similarity: {avg_similarity:.4f}")
    logger.info(f"  Min Similarity: {np.min(similarities):.4f}" if similarities else "  No samples")
    logger.info(f"  Max Similarity: {np.max(similarities):.4f}" if similarities else "")
    
    return results

def load_test_data() -> List[Dict]:
    """Load test data."""
    test_path = Path("data/processed/test_split.jsonl")
    
    if not test_path.exists():
        logger.info("Test data not found, using full dataset")
        data_path = Path("data/processed/train.jsonl")
    else:
        data_path = test_path
    
    if not data_path.exists():
        logger.error(f"No data found at {data_path}")
        logger.info("Please run: python data/download_data.py")
        return []
    
    data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    logger.info(f"Loaded {len(data)} test samples")
    return data

def save_results(results: Dict[str, Any], model_path: str):
    """Save evaluation results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{timestamp}.json"
    filepath = RESULTS_DIR / filename
    
    results["metadata"] = {
        "model_path": model_path,
        "timestamp": timestamp,
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"
    }
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {filepath}")
    return filepath

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Earnings Call Intelligence Model")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum samples to evaluate")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("EARNINGS CALL INTELLIGENCE - EVALUATION")
    logger.info("=" * 60)
    
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    pipe = create_pipeline(model, tokenizer)
    
    test_data = load_test_data()
    
    if not test_data:
        logger.error("No test data available")
        return
    
    results = {}
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    beat_miss_results = evaluate_beat_miss(pipe, tokenizer, test_data, args.max_samples)
    results["beat_miss"] = beat_miss_results
    
    quality_results = evaluate_response_quality(pipe, tokenizer, test_data, min(args.max_samples // 2, 20))
    results["response_quality"] = quality_results
    
    results["summary"] = {
        "beat_miss_accuracy": beat_miss_results.get("accuracy", 0),
        "f1_score": beat_miss_results.get("f1_score", 0),
        "response_quality": quality_results.get("avg_cosine_similarity", 0),
    }
    
    save_results(results, args.model_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info("\nSummary:")
    logger.info(f"  Beat/Miss Accuracy: {results['summary']['beat_miss_accuracy']:.4f}")
    logger.info(f"  F1 Score: {results['summary']['f1_score']:.4f}")
    logger.info(f"  Response Quality: {results['summary']['response_quality']:.4f}")
    
    return results

if __name__ == "__main__":
    main()
