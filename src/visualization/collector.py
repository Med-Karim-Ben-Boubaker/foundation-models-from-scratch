"""Clean collector service for generation data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GenerationStep:
    """Container for one generation step."""

    step: int
    topk_logits: torch.Tensor  # [K] - only top-k logits
    topk_probs: torch.Tensor  # [K] - only top-k probs
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
        topk_logits: torch.Tensor,
        topk_probs: torch.Tensor,
        next_id: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_vals: torch.Tensor,
    ) -> None:
        """Store CPU copies of generation step data."""
        # Ensure we're working with 1D tensors
        if topk_logits.dim() > 1:
            topk_logits = topk_logits.squeeze(0)
        if topk_probs.dim() > 1:
            topk_probs = topk_probs.squeeze(0)
        if topk_idx.dim() > 1:
            topk_idx = topk_idx.squeeze(0)
        if topk_vals.dim() > 1:
            topk_vals = topk_vals.squeeze(0)

        # Calculate metrics from top-k values only
        entropy = -(topk_probs * torch.log(topk_probs + 1e-8)).sum().item()
        max_logit = topk_logits.max().item()
        min_logit = topk_logits.min().item()

        self.steps.append(
            GenerationStep(
                step=step,
                topk_logits=topk_logits.detach().cpu(),
                topk_probs=topk_probs.detach().cpu(),
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
        # For next_token_id probability, we need to find it in topk_vals
        probs = []
        for step in self.steps:
            # Find the probability of the next token in the top-k values
            token_idx = (step.topk_idx == step.next_token_id).nonzero(as_tuple=True)[0]
            if len(token_idx) > 0:
                prob = step.topk_vals[token_idx[0]].item()
            else:
                # If next_token_id is not in top-k, we can't get its exact probability
                prob = 0.0
            probs.append(prob)

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
            "prompt_ids": self.prompt_ids.tolist()
            if self.prompt_ids is not None
            else None,
            "steps": [
                {
                    "step": step.step,
                    "topk_logits": step.topk_logits.tolist(),
                    "topk_probs": step.topk_probs.tolist(),
                    "next_token_id": step.next_token_id,
                    "topk_idx": step.topk_idx.tolist(),
                    "topk_vals": step.topk_vals.tolist(),
                    "entropy": step.entropy,
                    "max_logit": step.max_logit,
                    "min_logit": step.min_logit,
                }
                for step in self.steps
            ],
            "metadata": self.metadata,
            "summary": self.get_summary(),
        }
