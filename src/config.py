from dataclasses import dataclass

@dataclass
class GPTConfig:
  vocab_size: int = 50257
  context_length: int = 256
  emb_dim: int = 768
  n_heads: int = 12
  n_layers: int = 12
  drop_rate: float = 0.1
  qkv_bias: bool = False

@dataclass
class TrainConfig:
  batch_size: int = 2
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