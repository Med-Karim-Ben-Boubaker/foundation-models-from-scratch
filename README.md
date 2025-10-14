# LLM from Scratch

A simple implementation of a GPT-style language model built from scratch using PyTorch.

## Features

- GPT-style transformer architecture
- Configurable model parameters
- Training with validation
- Training history visualization

## Quick Start

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run training:
   ```bash
   uv run -m scripts.train
   ```

## Project Structure

- `scripts/train.py` - Main training script
- `src/` - Source code modules
  - `models/` - GPT model implementation
  - `data/` - Data loading and tokenization
  - `training/` - Training utilities and evaluation
  - `utils/` - Logging and other utilities
- `configs/` - Model and training configurations
- `data/` - Training data

## Configuration

The model uses YAML configuration files. Default configuration is in `configs/gpt_124m.yaml` with:
- 124M parameters
- 128 context length
- 6 transformer layers
- 8 attention heads

## Requirements

- Python >= 3.11
- PyTorch >= 2.8.0
- See `pyproject.toml` for full dependencies