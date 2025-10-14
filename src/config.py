from dataclasses import dataclass

# 30M parameters
@dataclass
class GPTConfig:
  vocab_size: int = 50257
  context_length: int = 128
  emb_dim: int = 256
  n_heads: int = 8
  n_layers: int = 6
  drop_rate: float = 0.1
  qkv_bias: bool = False

# 162M parameters
# @dataclass
# class GPTConfig:
#   vocab_size: int = 50257
#   context_length: int = 256
#   emb_dim: int = 768
#   n_heads: int = 12
#   n_layers: int = 12
#   drop_rate: float = 0.1
#   qkv_bias: bool = False

@dataclass
class TrainConfig:
  batch_size: int = 4
  lr: float = 4e-4
  weight_decay: float = 0.01
  num_epochs: int = 10
  eval_freq: int = 100
  eval_iter: int = 50
  grad_accum_steps: int = 1
  amp: bool = True
  device: str = "cuda"
  seed: int = 123
  num_workers: int = 0
  warmup_steps: int = 1450
  min_lr: float = 1e-5
  betas: tuple = (0.9, 0.999)
  eps: float = 1e-8
  fused: bool = True
  grad_clip_norm: float = 1.0