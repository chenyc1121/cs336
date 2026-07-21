from __future__ import annotations

from collections.abc import Sequence

import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token from a one-dimensional next-token logit vector."""
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

        # Keep the first token whose inclusion makes cumulative probability
        # reach top_p; discard only tokens after that point.
        remove = cumulative_probabilities - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
        sorted_probabilities /= sorted_probabilities.sum()

        sampled_position = torch.multinomial(sorted_probabilities, 1, generator=generator)
        return int(sorted_indices[sampled_position].item())

    return int(torch.multinomial(probabilities, 1, generator=generator).item())


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Generate completion token IDs, including the end token if sampled."""
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must have at least one parameter") from error

    context_length = model.context_length
    token_ids = list(prompt_ids)
    completion_ids: list[int] = []
    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            context = token_ids[-context_length:]
            inputs = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            next_token = sample_next_token(
                model(inputs)[0, -1],
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            token_ids.append(next_token)
            completion_ids.append(next_token)
            if next_token == end_token_id:
                break
    finally:
        model.train(was_training)

    return completion_ids
