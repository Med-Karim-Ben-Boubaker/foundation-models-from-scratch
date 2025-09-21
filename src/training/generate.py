from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from src.utils.logging import get_logger

from src.visualization.collector import GenerationTrace, GenerationStep

logger = get_logger(__name__)


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
    """
    Prevent repetition of n-grams by banning tokens that would complete repeated sequences.

    Algorithm: Build a map of (n-1)-gram prefixes to their possible next tokens,
    then ban tokens that would complete any previously seen n-gram.
    """
    if no_repeat_ngram_size <= 0:
        return logits

    seq = generated.squeeze(0).tolist()
    if len(seq) < no_repeat_ngram_size - 1:
        return logits

    prefix_len = no_repeat_ngram_size - 1
    next_for_prefix: Dict[Tuple[int, ...], List[int]] = {}

    # Build map of (n-1)-gram prefixes to their possible next tokens
    for i in range(len(seq) - no_repeat_ngram_size + 1):
        prefix = tuple(seq[i : i + prefix_len])
        nxt = seq[i + prefix_len]
        next_for_prefix.setdefault(prefix, []).append(nxt)

    # Get current (n-1)-gram prefix and ban its possible next tokens
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
    """
    Apply nucleus (top-p) and top-k filtering to logits.

    Top-k: Keep only the k highest probability tokens.
    Top-p: Keep tokens whose cumulative probability mass is <= p.
    """
    if top_k > 0:
        # Find the k-th largest value to use as threshold
        kth_vals = torch.topk(logits, k=min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = torch.where(
            logits < kth_vals, torch.full_like(logits, float("-inf")), logits
        )

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        # Shift mask right to keep first token above threshold (nucleus sampling)
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        # Convert sorted mask back to original logits order
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
    trace: bool = False,
    verbose: bool = False,
    topk_log: int = 10,
) -> Tuple[torch.Tensor, dict]:
    from src.visualization.printer import print_generation_step
    from src.data.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    out = input_ids
    new_count = 0

    if trace:
        collector = GenerationTrace()
        collector.set_prompt(out)
        collector.set_metadata(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
            }
        )

    if verbose:
        logger.info("=== Starting Generation ===")
        prompt_text = tokenizer.decode(input_ids[0].tolist())
        logger.info(f"Prompt: '{prompt_text}'")

    for step in range(max_new_tokens):
        # Sliding window: only use last context_size tokens to avoid memory issues
        context_tokens = out[:, -context_size:]

        # Get model predictions for next token
        logits = model(context_tokens)[:, -1, :]

        # Apply temperature scaling (higher temp = more randomness)
        if temperature > 0:
            logits = logits / temperature

        # Apply generation constraints in sequence
        logits = _apply_repetition_penalty(logits, out, repetition_penalty)
        logits = _enforce_no_repeat_ngram(logits, out, no_repeat_ngram_size)
        logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        # Convert to probabilities and sample next token
        probs = F.softmax(logits, dim=-1)
        if temperature > 0:
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding when temperature = 0
            next_id = torch.argmax(probs, dim=-1, keepdim=True)

        # Calculate top-k values once for both tracing and verbose printing
        k = min(topk_log, probs.shape[-1])
        topv, topi = torch.topk(probs, k=k, dim=-1)
        # Extract corresponding logits and probs for top-k tokens only
        topk_logits = logits.gather(dim=-1, index=topi)
        topk_probs = probs.gather(dim=-1, index=topi)

        # Calculate generation metrics from full distribution
        probs_1d = probs.squeeze(0) if probs.dim() > 1 else probs
        logits_1d = logits.squeeze(0) if logits.dim() > 1 else logits
        entropy = -(probs_1d * torch.log(probs_1d + 1e-8)).sum().item()
        max_logit = logits_1d.max().item()
        min_logit = logits_1d.min().item()

        if trace:
            collector.add_step(
                step + 1,
                topk_logits,
                topk_probs,
                next_id,
                topi,
                topv,
            )

        # Print step information if verbose
        if verbose:
            # Create a temporary step object for printing using already calculated values
            temp_step = GenerationStep(
                step=step + 1,
                topk_logits=topk_logits,
                topk_probs=topk_probs,
                next_token_id=int(next_id.item()),
                topk_idx=topi,
                topk_vals=topv,
                entropy=entropy,
                max_logit=max_logit,
                min_logit=min_logit,
            )
            print_generation_step(temp_step, tokenizer, top_k=min(5, k))

        # Append new token to sequence
        out = torch.cat([out, next_id], dim=1)
        new_count += 1

        # Early stopping on EOS token (but respect minimum length)
        if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
            if new_count >= min_new_tokens:
                if verbose:
                    logger.info(f"\nStopping early at step {step + 1} (EOS token)")
                break

    if verbose:
        final_text = tokenizer.decode(out[0].tolist())
        logger.info("\n=== Final Generated Text ===")
        logger.info(f"'{final_text}'")

    if trace:
        return out, collector.to_dict()
    else:
        return out, {}


def generate_text_simple(model, input_token_ids, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        context_tokens = input_token_ids[:, -context_size:]

        with torch.no_grad():
            logits = model(context_tokens)[:, -1, :]

            next_token_id = torch.argmax(
                torch.softmax(logits, dim=-1), dim=-1, keepdim=True
            )

            input_token_ids = torch.cat((input_token_ids, next_token_id), dim=1)

    return input_token_ids
