"""Generate a completion from a trained CS336 Transformer checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.bpe import Tokenizer
from cs336_basics.decoding import generate
from cs336_basics.transformer import transformer


END_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=Path("artifacts/tinystories_bpe/vocab.json"))
    parser.add_argument("--merges", type=Path, default=Path("artifacts/tinystories_bpe/merges.json"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--rmsnorm-eps", type=float, default=1e-5)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, [END_TOKEN])
    end_token_id = next(
        token_id for token_id, token_bytes in tokenizer.vocab.items() if token_bytes == END_TOKEN.encode()
    )

    model = transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        eps=args.rmsnorm_eps,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    prompt_ids = tokenizer.encode(args.prompt)
    # An empty prompt starts a new document from the corpus boundary token.
    model_prompt_ids = prompt_ids or [end_token_id]
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    completion_ids = generate(
        model,
        model_prompt_ids,
        args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        end_token_id=end_token_id,
        generator=generator,
    )
    if completion_ids and completion_ids[-1] == end_token_id:
        completion_ids.pop()
    print(args.prompt + tokenizer.decode(completion_ids))


if __name__ == "__main__":
    main()
