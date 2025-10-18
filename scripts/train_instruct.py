#!/usr/bin/env python3
"""
Training script for instruction fine-tuning.

This script loads a pre-trained model and fine-tunes it on instruction-following data.
"""

import os
import json
import torch
import yaml
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from src.config import GPTConfig, TrainConfig
from src.models.gpt import GPTModel
from src.data.dataloader import create_instruction_dataloader
from src.training.trainer import train_instruction_finetuning
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_instruction_data(file_path: str, max_examples: int = None) -> list:
    with open(file_path, 'r') as f:
        examples = json.load(f)
    
    if max_examples is not None:
        examples = examples[:max_examples]
    
    logger.info(f"Loaded {len(examples)} instruction examples from {file_path}")
    return examples


def split_instruction_data(examples: list, train_ratio: float = 0.9) -> tuple:
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    logger.info(f"Split data: {len(train_examples)} train, {len(val_examples)} validation")
    return train_examples, val_examples


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Configuration - use fine-tuning specific config
    cfg_path = os.environ.get("CFG", "configs/gpt2_35m_4heads_12layers_finetuning.yaml")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    gcfg = GPTConfig(**raw["model"])
    tcfg = TrainConfig(**raw["train"])

    # Check if CUDA is available (Necessary for training)
    device = torch.device(tcfg.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA is not available. Training will be slow.")
        raise RuntimeError("CUDA is not available. Please check your GPU configuration.")

    logger.info(f"Using device: {device}")

    # Setup TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"instruct_finetuning_{timestamp}"
    tensorboard_log_dir = f"runs/{run_name}"
    
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logger.info(f"TensorBoard logging to: {tensorboard_log_dir}")
    
    # Log hyperparameters to TensorBoard
    writer.add_hparams(
        {
            "learning_rate": tcfg.lr,
            "batch_size": tcfg.batch_size,
            "num_epochs": tcfg.num_epochs,
            "grad_accum_steps": tcfg.grad_accum_steps,
            "eval_freq": tcfg.eval_freq,
            "context_length": gcfg.context_length,
            "n_heads": gcfg.n_heads,
            "n_layers": gcfg.n_layers,
            "emb_dim": gcfg.emb_dim,
            "training_type": "instruction_finetuning",
        },
        {}
    )

    # Load instruction data
    instruction_data_path = "data/alpaca_data.json"
    max_examples = None
    
    examples = load_instruction_data(instruction_data_path, max_examples)
    train_examples, val_examples = split_instruction_data(examples)

    # Create data loaders
    train_loader = create_instruction_dataloader(
        examples=train_examples,
        batch_size=tcfg.batch_size,
        max_length=gcfg.context_length,
        shuffle=True,
        drop_last=True,
        num_workers=tcfg.num_workers,
    )
    
    val_loader = create_instruction_dataloader(
        examples=val_examples,
        batch_size=tcfg.batch_size,
        max_length=gcfg.context_length,
        shuffle=False,
        drop_last=False,
        num_workers=tcfg.num_workers,
    )

    # Load pre-trained model from artifacts folder
    model_checkpoint_path = "artifacts/gpt2_35m_4heads_12layers.pth"
    
    if os.path.exists(model_checkpoint_path):
        logger.info(f"Loading pre-trained model from {model_checkpoint_path}")
        model = GPTModel(gcfg)
        model_checkpoint = torch.load(model_checkpoint_path, map_location=device)
        model.load_state_dict(model_checkpoint["model"])
        logger.info("Pre-trained model loaded successfully")
    else:
        logger.error(f"Pre-trained model not found at {model_checkpoint_path}")
        logger.error("Please ensure you have a trained model in the artifacts folder before running instruction fine-tuning")
        raise FileNotFoundError(f"Pre-trained model not found at {model_checkpoint_path}")
    
    model = model.to(device)

    logger.info("Starting instruction fine-tuning...")
    _, _, train_loss, val_loss, step_numbers, optimizer = train_instruction_finetuning(
        model, train_loader, val_loader, device, tcfg, writer=writer
    )

    # Save fine-tuned model with _finetuned suffix
    os.makedirs("artifacts", exist_ok=True)
    
    # Extract base model name and add _finetuned suffix
    base_model_name = "gpt2_35m_4heads_12layers"
    model_path = os.path.join("artifacts", f"{base_model_name}_finetuned.pth")
    
    logger.info(f"Saving fine-tuned model to {model_path}")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": gcfg.__dict__,
            "training_config": tcfg.__dict__,
            "timestamp": timestamp,
            "base_model": base_model_name,
            "fine_tuning_type": "instruction_finetuning",
        },
        model_path,
    )
    logger.info(f"Fine-tuned model saved to {model_path}")
    
    # Close TensorBoard writer
    writer.close()
    logger.info("TensorBoard logging completed")


if __name__ == "__main__":
    main()
