"""
Training Configuration for Earnings Call Intelligence System
============================================================

This file contains all hyperparameters and settings for QLoRA fine-tuning.
Optimized for Google Colab T4 GPU (free tier).
"""

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "unsloth/Llama-3.2-3B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[str] = None

@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head"
    ])
    
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = True
    use_dora: bool = True

@dataclass
class TrainingConfig:
    """Training configuration optimized for T4 GPU."""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    
    effective_batch_size: int = 16
    
    warmup_ratio: float = 0.03
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    
    lr_scheduler_type: str = "cosine_with_restarts"
    num_cycles: int = 3
    
    optim: str = "paged_adamw_8bit"
    group_by_length: bool = True
    
    eval_strategy: str = "steps"
    eval_steps: int = 250
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    logging_strategy: str = "steps"
    logging_steps: int = 50
    report_to: str = "tensorboard"
    
    fp16: bool = False
    bf16: bool = True
    
    dataloader_num_workers: int = 4
    packing: bool = True
    
    seed: int = 42
    remove_unused_columns: bool = False

@dataclass
class DataConfig:
    """Data configuration."""
    train_file: str = "data/processed/train.jsonl"
    eval_file: Optional[str] = None
    test_file: str = "data/processed/test.jsonl"
    
    text_field: str = "text"
    dataset_text_field: str = "messages"
    
    train_split: float = 0.8
    eval_split: float = 0.1
    test_split: float = 0.1

@dataclass
class OutputConfig:
    """Output configuration."""
    output_dir: str = "training/outputs"
    run_name: str = "earnings-intelligence-v1"
    logging_dir: str = "training/logs"

@dataclass
class FullConfig:
    """Full configuration combining all settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def save(self, path: str):
        """Save configuration to file."""
        import json
        
        config_dict = {
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "output": self.output.__dict__
        }
        
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "FullConfig":
        """Load configuration from file."""
        import json
        
        with open(path, "r") as f:
            config_dict = json.load(f)
        
        config = cls()
        config.model = ModelConfig(**config_dict["model"])
        config.lora = LoRAConfig(**config_dict["lora"])
        config.training = TrainingConfig(**config_dict["training"])
        config.data = DataConfig(**config_dict["data"])
        config.output = OutputConfig(**config_dict["output"])
        
        return config

def get_default_config() -> FullConfig:
    """Get default configuration."""
    return FullConfig()

def print_config(config: FullConfig):
    """Print configuration in a readable format."""
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    
    print("\n📦 Model:")
    for key, value in config.model.__dict__.items():
        print(f"   {key}: {value}")
    
    print("\n🔧 LoRA:")
    for key, value in config.lora.__dict__.items():
        print(f"   {key}: {value}")
    
    print("\n🎯 Training:")
    for key, value in config.training.__dict__.items():
        print(f"   {key}: {value}")
    
    print("\n📁 Data:")
    for key, value in config.data.__dict__.items():
        print(f"   {key}: {value}")
    
    print("\n📤 Output:")
    for key, value in config.output.__dict__.items():
        print(f"   {key}: {value}")
    
    print("=" * 60)
