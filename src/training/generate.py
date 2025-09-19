from typing import Optional, Tuple
import torch
import torch.nn.functional as F

def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering."""
    if top_k > 0:
        kth_vals = torch.topk(logits, k=min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)

    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        # Shift mask right to keep first token above threshold
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

def generate_text_simple(model, input_token_ids, max_new_tokens, context_size):
    """
    Generate new text tokens using a simple greedy decoding strategy.

    Args:
        model: The trained language model
        input_token_ids: Starting sequence of token IDs (batch_size, seq_len)
        max_new_tokens: Number of new tokens to generate
        context_size: Maximum context window size for the model

    Returns:
        Complete sequence including original input and generated tokens
    """
    for _ in range(max_new_tokens):
        # Extract only the last context_size tokens to stay within model's context window
        context_tokens = input_token_ids[:, -context_size:]

        # Disable gradient computation for inference (faster and uses less memory)
        with torch.no_grad():
            # Get model predictions for the context tokens
            logits = model(context_tokens)[
                :, -1, :
            ]  # Only use the last position's logits

            # Select the most likely next token using greedy decoding (argmax)
            next_token_id = torch.argmax(
                torch.softmax(logits, dim=-1), dim=-1, keepdim=True
            )

            # Append the new token to the sequence
            input_token_ids = torch.cat((input_token_ids, next_token_id), dim=1)

    return input_token_ids
