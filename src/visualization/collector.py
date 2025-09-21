"""Clean collector service for generation data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GenerationStep:
    """Container for one generation step."""
    step: int
    logits: torch.Tensor  # [V]
    probs: torch.Tensor   # [V]
    next_token_id: int
    topk_idx: torch.Tensor  # [K]
    topk_vals: torch.Tensor  # [K]
    entropy: float
    max_logit: float
    min_logit: float


@dataclass
class GenerationTrace:
    """Accumulates generation details for analysis."""
    steps: List[GenerationStep] = field(default_factory=list)
    prompt_ids: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_prompt(self, prompt_ids: torch.Tensor) -> None:
        """Record the prompt token ids (CPU copy)."""
        self.prompt_ids = prompt_ids.detach().cpu()

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Record the metadata."""
        self.metadata = metadata

    def add_step(
        self,
        step: int,
        logits: torch.Tensor,
        probs: torch.Tensor,
        next_id: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_vals: torch.Tensor,
    ) -> None:
        """Store CPU copies of generation step data."""
        # Ensure we're working with 1D tensors
        if logits.dim() > 1:
            logits = logits.squeeze(0)
        if probs.dim() > 1:
            probs = probs.squeeze(0)
        if topk_idx.dim() > 1:
            topk_idx = topk_idx.squeeze(0)
        if topk_vals.dim() > 1:
            topk_vals = topk_vals.squeeze(0)

        # Calculate metrics
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        max_logit = logits.max().item()
        min_logit = logits.min().item()

        self.steps.append(
            GenerationStep(
                step=step,
                logits=logits.detach().cpu(),
                probs=probs.detach().cpu(),
                next_token_id=int(next_id.item()),
                topk_idx=topk_idx.detach().cpu(),
                topk_vals=topk_vals.detach().cpu(),
                entropy=entropy,
                max_logit=max_logit,
                min_logit=min_logit,
            )
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the generation trace."""
        if not self.steps:
            return {"num_steps": 0, "avg_entropy": 0.0, "avg_prob": 0.0}
        
        entropies = [step.entropy for step in self.steps]
        probs = [step.probs[step.next_token_id].item() for step in self.steps]
        
        return {
            "num_steps": len(self.steps),
            "avg_entropy": sum(entropies) / len(entropies),
            "avg_prob": sum(probs) / len(probs),
            "min_entropy": min(entropies),
            "max_entropy": max(entropies),
            "min_prob": min(probs),
            "max_prob": max(probs),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "prompt_ids": self.prompt_ids,
            "steps": [
                {
                    "step": step.step,
                    "logits": step.logits,
                    "probs": step.probs,
                    "next_token_id": step.next_token_id,
                    "topk_idx": step.topk_idx,
                    "topk_vals": step.topk_vals,
                    "entropy": step.entropy,
                    "max_logit": step.max_logit,
                    "min_logit": step.min_logit,
                }
                for step in self.steps
            ],
            "metadata": self.metadata,
            "summary": self.get_summary(),
        }
