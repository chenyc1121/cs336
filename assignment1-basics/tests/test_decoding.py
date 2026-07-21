import pytest
import torch

from cs336_basics.decoding import generate, sample_next_token


class FixedLogitModel(torch.nn.Module):
    def __init__(self, next_token: int, context_length: int = 3, vocab_size: int = 5):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.next_token = next_token
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.seen_lengths: list[int] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.seen_lengths.append(inputs.shape[-1])
        logits = torch.full((*inputs.shape, self.vocab_size), -100.0, device=inputs.device)
        logits[..., self.next_token] = 0.0 + self.anchor
        return logits


def test_temperature_scaling_before_sampling():
    logits = torch.tensor([0.0, 1.0, 2.0])
    expected_generator = torch.Generator().manual_seed(7)
    expected = torch.multinomial(torch.softmax(logits / 0.5, dim=-1), 1, generator=expected_generator).item()

    actual_generator = torch.Generator().manual_seed(7)
    actual = sample_next_token(logits, temperature=0.5, generator=actual_generator)
    assert actual == expected


def test_top_p_discards_tokens_after_nucleus():
    logits = torch.log(torch.tensor([0.7, 0.2, 0.1]))
    generator = torch.Generator().manual_seed(1)
    samples = [sample_next_token(logits, top_p=0.6, generator=generator) for _ in range(20)]
    assert samples == [0] * 20


def test_generate_stops_at_end_token_and_restores_mode():
    model = FixedLogitModel(next_token=4)
    model.train()
    completion = generate(model, [1, 2], 10, end_token_id=4)
    assert completion == [4]
    assert model.training


def test_generate_limits_tokens_and_crops_context():
    model = FixedLogitModel(next_token=2, context_length=3)
    completion = generate(model, [0, 1, 2, 3], 4)
    assert completion == [2, 2, 2, 2]
    assert model.seen_lengths == [3, 3, 3, 3]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
    ],
)
def test_sample_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        sample_next_token(torch.ones(3), **kwargs)
