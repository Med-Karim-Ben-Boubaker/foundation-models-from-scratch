from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


def _apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """
    Penalize tokens that have appeared at least once in the generated sequence.
    """
    if penalty <= 1.0 or generated.numel() == 0:
        return logits

    unique_tokens = torch.unique(generated)
    token_indices = unique_tokens.long()

    vals = logits[0, token_indices]

    pos_mask = vals > 0
    # positive logits get divided, negative get multiplied
    vals = torch.where(pos_mask, vals / penalty, vals * penalty)

    logits[0, token_indices] = vals

    return logits

def _enforce_no_repeat_ngram(
    logits: torch.Tensor, generated: torch.Tensor, no_repeat_ngram_size: int
) -> torch.Tensor:

    if no_repeat_ngram_size <= 0:
        return logits

    seq = generated.squeeze(0).tolist()
    if len(seq) < no_repeat_ngram_size - 1:
        return logits

    prefix_len = no_repeat_ngram_size - 1
    next_for_prefix: Dict[Tuple[int, ...], List[int]] = {}
    for i in range(len(seq) - no_repeat_ngram_size + 1):
        prefix = tuple(seq[i : i + prefix_len])
        nxt = seq[i + prefix_len]
        next_for_prefix.setdefault(prefix, []).append(nxt)

    cur_prefix = tuple(seq[-prefix_len:])
    banned = next_for_prefix.get(cur_prefix, [])
    if banned:
        banned_idx = torch.tensor(
            banned, device=logits.device, dtype=torch.long
        ).unsqueeze(0)
        logits = logits.scatter(dim=-1, index=banned_idx, value=float("-inf"))
    return logits


def _top_k_top_p_filtering(
    logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0
) -> torch.Tensor:

    if top_k > 0:
        kth_vals = torch.topk(logits, k=min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = torch.where(
            logits < kth_vals, torch.full_like(logits, float("-inf")), logits
        )

    if 0.0 < top_p < 1.0:
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

    return logits


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    *,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
    eos_token_id: Optional[int] = None,
    min_new_tokens: int = 8,
) -> torch.Tensor:

    out = input_ids
    new_count = 0

    for _ in range(max_new_tokens):
        context_tokens = out[:, -context_size:]

        logits = model(context_tokens)[:, -1, :]

        if temperature > 0:
            logits = logits / temperature

        logits = _apply_repetition_penalty(logits, out, repetition_penalty)

        logits = _enforce_no_repeat_ngram(logits, out, no_repeat_ngram_size)

        logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, dim=-1)
        if temperature > 0:
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(probs, dim=-1, keepdim=True)

        out = torch.cat([out, next_id], dim=1)
        new_count += 1

        if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
            if new_count >= min_new_tokens:
                break

    return out


def generate_text_simple(model, input_token_ids, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        context_tokens = input_token_ids[:, -context_size:]

        with torch.no_grad():
            logits = model(context_tokens)[
                :, -1, :
            ] 

            next_token_id = torch.argmax(
                torch.softmax(logits, dim=-1), dim=-1, keepdim=True
            )

            input_token_ids = torch.cat((input_token_ids, next_token_id), dim=1)

    return input_token_ids
