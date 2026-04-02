"""
Data Preparation for Earnings Call Intelligence System
=======================================================

Prepares and formats data for instruction tuning with QLoRA.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_conversation(messages: List[Dict[str, str]], tokenizer, max_length: int = 2048) -> Dict[str, Any]:
    """
    Format a conversation into training-ready text.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        
    Returns:
        Dict with 'text' key containing formatted string
    """
    if isinstance(messages[0], dict) and "messages" in messages[0]:
        messages = messages[0]["messages"]
    
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = format_conversation_fallback(messages)
    else:
        text = format_conversation_fallback(messages)
    
    if len(text) > max_length:
        text = text[:max_length]
    
    return {"text": text}

def format_conversation_fallback(messages: List[Dict[str, str]]) -> str:
    """Fallback formatting when tokenizer doesn't support chat template."""
    formatted = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        formatted.append(f"<|{role}|>\n{content}")
    return "\n".join(formatted) + "\n<|ASSISTANT|>"

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def prepare_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 2048,
    split: str = "train"
) -> Dataset:
    """
    Prepare dataset for training.
    
    Args:
        data_path: Path to JSONL data file
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        split: Dataset split name
        
    Returns:
        HuggingFace Dataset
    """
    logger.info(f"Loading data from {data_path}")
    
    try:
        data = load_jsonl(data_path)
        logger.info(f"Loaded {len(data)} examples")
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run: python data/download_data.py")
        raise
    
    formatted_data = []
    for item in data:
        formatted = format_conversation(item, tokenizer, max_length)
        formatted["raw"] = item
        formatted_data.append(formatted)
    
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.select(range(min(len(dataset), 10000)))
    
    logger.info(f"Prepared {len(dataset)} examples for {split}")
    return dataset

def create_test_set(
    all_data: List[Dict],
    tokenizer,
    test_ratio: float = 0.1,
    max_length: int = 2048
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create train and test splits.
    
    Args:
        all_data: All available data
        tokenizer: Tokenizer for encoding
        test_ratio: Ratio of data for testing
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_data, test_data)
    """
    import random
    random.seed(42)
    
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * (1 - test_ratio))
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test")
    
    return train_data, test_data

def augment_data(data: List[Dict]) -> List[Dict]:
    """
    Augment training data with variations.
    
    Args:
        data: Original training data
        
    Returns:
        Augmented data
    """
    augmented = []
    
    variations = [
        ("Analyze the sentiment", "What is the sentiment of"),
        ("Predict whether they beat", "Did they beat or miss"),
        ("Extract the key metrics", "What were the main financial metrics"),
    ]
    
    for item in data:
        augmented.append(item)
        
        if isinstance(item, dict) and "messages" in item:
            messages = item["messages"]
            if len(messages) >= 2:
                user_content = messages[1]["content"]
                assistant_content = messages[2]["content"]
                
                for old, new in variations:
                    if old in user_content:
                        new_item = {
                            "messages": [
                                messages[0],
                                {"role": "user", "content": user_content.replace(old, new)},
                                {"role": "assistant", "content": assistant_content}
                            ]
                        }
                        augmented.append(new_item)
                        break
    
    logger.info(f"Augmented data: {len(data)} -> {len(augmented)} examples")
    return augmented

def validate_data(data: List[Dict]) -> bool:
    """
    Validate data format.
    
    Args:
        data: Data to validate
        
    Returns:
        True if valid, False otherwise
    """
    for i, item in enumerate(data):
        if isinstance(item, dict) and "messages" in item:
            messages = item["messages"]
            
            if not isinstance(messages, list):
                logger.error(f"Item {i}: messages is not a list")
                return False
            
            if len(messages) < 2:
                logger.error(f"Item {i}: not enough messages")
                return False
            
            for j, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    logger.error(f"Item {i}, message {j}: not a dict")
                    return False
                
                if "content" not in msg:
                    logger.error(f"Item {i}, message {j}: missing content")
                    return False
    
    return True

def main():
    """Test data preparation."""
    from transformers import AutoTokenizer
    
    logger.info("Testing data preparation...")
    
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    data_path = "data/processed/train.jsonl"
    
    if Path(data_path).exists():
        dataset = prepare_dataset(data_path, tokenizer)
        print(f"\nDataset prepared: {len(dataset)} examples")
        print(f"First example length: {len(dataset[0]['text'])} chars")
    else:
        logger.warning(f"Data not found at {data_path}")
        logger.info("Run: python data/download_data.py")

if __name__ == "__main__":
    main()
