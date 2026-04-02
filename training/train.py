"""
Main Training Script for Earnings Call Intelligence System
==========================================================

Fine-tunes Llama-3.2-3B-Instruct using QLoRA for earnings call analysis.

Usage:
    python training/train.py
    
Requirements:
    - NVIDIA GPU (T4 recommended, works on Google Colab free tier)
    - ~6 hours of GPU time
    - 10GB+ free disk space
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

from train_config import get_default_config, print_config, FullConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability and properties."""
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Training requires GPU.")
        logger.info("On Google Colab: Runtime > Change runtime > T4 GPU")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info("=" * 60)
    logger.info("GPU DETECTED")
    logger.info("=" * 60)
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"VRAM: {gpu_memory:.2f} GB")
    logger.info("=" * 60)
    
    if gpu_memory < 8:
        logger.warning("Low VRAM. Consider using smaller batch size.")
    
    return True

def load_model_and_tokenizer(config: FullConfig):
    """Load model with QLoRA configuration."""
    logger.info(f"Loading model: {config.model.model_name}")
    
    try:
        from unsloth import FastLanguageModel
        
        logger.info("Using unsloth for optimized loading (2x faster training)")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model.model_name,
            max_seq_length=config.model.max_seq_length,
            load_in_4bit=config.model.load_in_4bit,
            dtype=None,
        )
        
        logger.info("Applying LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            use_gradient_checkpointing=config.lora.use_gradient_checkpointing,
            use_rslora=config.lora.use_rslora,
            use_dora=config.lora.use_dora,
            task_type=config.lora.task_type,
        )
        
    except ImportError:
        logger.warning("unsloth not available, using standard transformers")
        
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        bnb_config = None
        if config.model.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )
        
        model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model, tokenizer

def prepare_training_data(config: FullConfig, tokenizer):
    """Prepare training and evaluation datasets."""
    from training.data_prep import prepare_dataset, load_jsonl
    
    data_path = config.data.train_file
    
    if not Path(data_path).exists():
        logger.info(f"Data not found at {data_path}")
        logger.info("Running data download...")
        os.system("python data/download_data.py")
    
    logger.info("Loading and preparing training data...")
    
    raw_data = load_jsonl(data_path)
    
    import random
    random.seed(42)
    random.shuffle(raw_data)
    
    split_idx_train = int(len(raw_data) * 0.8)
    split_idx_eval = int(len(raw_data) * 0.9)
    
    train_data = raw_data[:split_idx_train]
    eval_data = raw_data[split_idx_train:split_idx_eval]
    test_data = raw_data[split_idx_eval:]
    
    train_path = Path("data/processed/train_split.jsonl")
    eval_path = Path("data/processed/eval_split.jsonl")
    test_path = Path("data/processed/test_split.jsonl")
    
    for path, data in [(train_path, train_data), (eval_path, eval_data), (test_path, test_data)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    
    def format_function(examples):
        texts = []
        for item in examples["raw"]:
            messages = item.get("messages", [])
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except:
                    continue
            else:
                text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
            texts.append(text)
        return {"text": texts}
    
    from datasets import Dataset
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    train_dataset = train_dataset.map(format_function, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_function, batched=True, remove_columns=eval_dataset.column_names)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

def create_trainer(model, tokenizer, train_dataset, eval_dataset, config: FullConfig):
    """Create and configure the SFTTrainer."""
    
    output_dir = f"{config.output.output_dir}/{config.output.run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("training/logs").mkdir(parents=True, exist_ok=True)
    
    training_args = SFTConfig(
        output_dir=output_dir,
        run_name=config.output.run_name,
        
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        
        warmup_ratio=config.training.warmup_ratio,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        
        lr_scheduler_type=config.training.lr_scheduler_type,
        num_cycles=config.training.num_cycles,
        
        optim=config.training.optim,
        group_by_length=config.training.group_by_length,
        
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        
        logging_dir=config.output.logging_dir,
        logging_strategy=config.training.logging_strategy,
        logging_steps=config.training.logging_steps,
        
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataset_text_field=config.data.text_field,
        packing=config.training.packing,
        
        remove_unused_columns=config.training.remove_unused_columns,
        seed=config.training.seed,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=config.model.max_seq_length,
    )
    
    return trainer

def train(config: FullConfig):
    """Main training function."""
    logger.info("=" * 60)
    logger.info("EARNINGS CALL INTELLIGENCE - TRAINING")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    logger.info("=" * 60)
    
    print_config(config)
    
    check_gpu()
    
    logger.info("\n--- Loading Model ---")
    model, tokenizer = load_model_and_tokenizer(config)
    
    logger.info("\n--- Preparing Data ---")
    train_dataset, eval_dataset = prepare_training_data(config, tokenizer)
    
    logger.info("\n--- Creating Trainer ---")
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, config)
    
    logger.info("\n--- Starting Training ---")
    logger.info("This will take approximately 4-6 hours on T4 GPU")
    logger.info("Press Ctrl+C to stop (progress will be saved)")
    
    try:
        train_result = trainer.train()
        
        logger.info("\n--- Training Complete ---")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        logger.info("\n--- Saving Model ---")
        trainer.save_model()
        trainer.save_state()
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        config_path = f"{config.output.output_dir}/{config.output.run_name}/config.json"
        config.save(config_path)
        logger.info(f"Training config saved to {config_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUCCESSFUL!")
        logger.info("=" * 60)
        logger.info(f"\nModel saved to: {config.output.output_dir}/{config.output.run_name}")
        logger.info(f"\nNext steps:")
        logger.info("  1. Evaluate: python evaluation/evaluate.py")
        logger.info("  2. Demo: python demo/gradio_app.py")
        logger.info("  3. Deploy: Upload to HuggingFace Hub")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_model(f"{config.output.output_dir}/checkpoint-interrupted")
        logger.info("Checkpoint saved. Resume training with --resume_from_checkpoint")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Earnings Call Intelligence Model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    args = parser.parse_args()
    
    config = get_default_config()
    
    if args.output_dir:
        config.output.output_dir = args.output_dir
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    
    train(config)

if __name__ == "__main__":
    main()
