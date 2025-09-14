import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
      logits = model(idx_cond)[:, -1, :]
      next_id = torch.argmax(torch.softmax(logits, dim=-1), dim=-1, keepdim=True)
      idx = torch.cat((idx, next_id), dim=1)
  return idx