import torch
import torch.nn as nn
from src.models.layers import TransformerBlock, LayerNorm

class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
    self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
    self.drop_emb = nn.Dropout(cfg.drop_rate)
    self.trf_blocks = nn.Sequential(*[
      TransformerBlock(cfg.emb_dim, cfg.context_length, cfg.n_heads, cfg.drop_rate, cfg.qkv_bias)
      for _ in range(cfg.n_layers)
    ])
    self.final_norm = LayerNorm(cfg.emb_dim)
    self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
    self.out_head.weight = self.tok_emb.weight

  def forward(self, in_idx):
    b, t = in_idx.shape
    x = self.tok_emb(in_idx) + self.pos_emb(torch.arange(t, device=in_idx.device))
    x = self.drop_emb(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    return self.out_head(x)