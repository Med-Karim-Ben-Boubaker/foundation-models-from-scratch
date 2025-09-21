"""Visualization printer service for generation traces."""

from typing import Optional
import torch
from src.data.tokenizer import get_tokenizer
from src.visualization.collector import GenerationStep, GenerationTrace


def print_generation_step(step: GenerationStep, tokenizer, top_k: int = 5) -> None:
    """Print information about one generation step."""
    print(f"\n--- Step {step.step} ---")
    
    probs = step.probs
    if probs.dim() > 1:
        probs = probs.squeeze(0)
    
    print(f"Chosen token ID: {step.next_token_id}")
    try:
        chosen_token = tokenizer.decode([step.next_token_id])
        print(f"Chosen token: '{chosen_token}'")
    except:
        print(f"Chosen token: <decode_error>")
    
    print(f"Chosen probability: {probs[step.next_token_id].item():.4f}")
    
    print(f"Top-{top_k} candidates:")
    # Ensure topk tensors are 1D
    topk_vals = step.topk_vals
    topk_idx = step.topk_idx
    if topk_vals.dim() > 1:
        topk_vals = topk_vals.squeeze(0)
    if topk_idx.dim() > 1:
        topk_idx = topk_idx.squeeze(0)
    
    for i, (prob, token_id) in enumerate(zip(topk_vals, topk_idx)):
        try:
            token = tokenizer.decode([token_id.item()])
            print(f"  {i+1}. ID {token_id.item():5d} | P={prob.item():.4f} | '{token}'")
        except:
            print(f"  {i+1}. ID {token_id.item():5d} | P={prob.item():.4f} | <decode_error>")
    
    print(f"Entropy: {step.entropy:.4f}")
    print(f"Max logit: {step.max_logit:.2f}")
    print(f"Min logit: {step.min_logit:.2f}")


def print_generation_summary(trace: GenerationTrace, tokenizer) -> None:
    """Print a summary of the generation trace."""
    if not trace.steps:
        print("No generation steps recorded.")
        return
    
    print("\n=== Generation Summary ===")
    summary = trace.get_summary()
    print(f"Total steps: {summary['num_steps']}")
    print(f"Average entropy: {summary['avg_entropy']:.4f}")
    print(f"Average chosen probability: {summary['avg_prob']:.4f}")
    print(f"Entropy range: {summary['min_entropy']:.4f} - {summary['max_entropy']:.4f}")
    print(f"Probability range: {summary['min_prob']:.4f} - {summary['max_prob']:.4f}")
    
    # Show final generated text
    if trace.prompt_ids is not None:
        prompt_ids = trace.prompt_ids
        if prompt_ids.dim() > 1:
            prompt_ids = prompt_ids.squeeze(0)
        prompt_text = tokenizer.decode(prompt_ids.tolist())
        print(f"\nPrompt: '{prompt_text}'")
    
    # Show generated tokens
    generated_tokens = []
    for step in trace.steps:
        try:
            token = tokenizer.decode([step.next_token_id])
            generated_tokens.append(token)
        except:
            generated_tokens.append(f"<{step.next_token_id}>")
    
    generated_text = "".join(generated_tokens)
    print(f"Generated: '{generated_text}'")


def print_all_steps(trace: GenerationTrace, tokenizer, top_k: int = 5) -> None:
    """Print all generation steps."""
    for step in trace.steps:
        print_generation_step(step, tokenizer, top_k)
