"""Train the CS336 Transformer language model.

Example (CPU):
    python train.py --train-data data/train.bin --valid-data data/valid.bin \
        --vocab-size 10000 --context-length 256 --device cpu

Example (GPU 0):
    python train.py --train-data data/train.bin --valid-data data/valid.bin \
        --vocab-size 10000 --context-length 256 --device cuda:0
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.adamw import adamW
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.coslr import coslr
from cs336_basics.crossentropy import crossentropy
from cs336_basics.dataloader import get_batch
from cs336_basics.grad_clip import grad_clip
from cs336_basics.transformer import transformer


DTYPES = {
    "uint16": np.uint16,
    "int32": np.int32,
    "int64": np.int64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Data and device.
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="uint16")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=42)

    # Model hyperparameters.
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--rmsnorm-eps", type=float, default=1e-5)

    # Optimization and schedule hyperparameters.
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--cosine-cycle-iters", type=int, default=None)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999), metavar=("BETA1", "BETA2"))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Logging and checkpointing.
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)

    args = parser.parse_args()
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iters
    if args.cosine_cycle_iters <= args.warmup_iters:
        parser.error("--cosine-cycle-iters must be greater than --warmup-iters")
    if args.max_iters <= 0 or args.batch_size <= 0 or args.context_length <= 0:
        parser.error("--max-iters, --batch-size, and --context-length must be positive")
    if args.eval_interval <= 0 or args.eval_iters <= 0 or args.log_interval <= 0:
        parser.error("--eval-interval, --eval-iters, and --log-interval must be positive")
    if args.d_model % args.num_heads != 0:
        parser.error("--d-model must be divisible by --num-heads")
    if args.resume is not None and args.checkpoint is None:
        args.checkpoint = args.resume
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
        )
    return device


def open_token_ids(path: Path, dtype_name: str) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=DTYPES[dtype_name], mode="r")


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        inputs, targets = get_batch(dataset, batch_size, context_length, str(device))
        losses.append(float(crossentropy(model(inputs), targets).item()))
    model.train()
    return float(np.mean(losses))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)

    train_data = open_token_ids(args.train_data, args.dtype)
    valid_data = open_token_ids(args.valid_data, args.dtype)
    for name, data in (("training", train_data), ("validation", valid_data)):
        if len(data) <= args.context_length:
            raise ValueError(f"The {name} dataset must contain more than context_length tokens")

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
    optimizer = adamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed checkpoint {args.resume} at iteration {start_iter}")

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"device={device}; train_tokens={len(train_data):,}; valid_tokens={len(valid_data):,}")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")

    start_time = time.perf_counter()
    for iteration in range(start_iter, args.max_iters):
        learning_rate = coslr(
            iteration,
            args.lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, str(device))
        optimizer.zero_grad(set_to_none=True)
        loss = crossentropy(model(inputs), targets)
        loss.backward()
        if args.max_grad_norm > 0:
            grad_clip(model.parameters(), args.max_grad_norm)
        optimizer.step()

        completed = iteration + 1
        if completed % args.log_interval == 0 or completed == 1:
            elapsed = time.perf_counter() - start_time
            print(
                f"iter {completed:>7d}/{args.max_iters} "
                f"loss={loss.item():.4f} lr={learning_rate:.3e} "
                f"tokens/s={args.batch_size * args.context_length * completed / max(elapsed, 1e-9):.0f}"
            )

        if completed % args.eval_interval == 0 or completed == args.max_iters:
            train_loss = estimate_loss(
                model, train_data, args.batch_size, args.context_length, args.eval_iters, device
            )
            valid_loss = estimate_loss(
                model, valid_data, args.batch_size, args.context_length, args.eval_iters, device
            )
            print(
                f"eval iter {completed:>7d}: train_loss={train_loss:.4f} "
                f"valid_loss={valid_loss:.4f} perplexity={math.exp(valid_loss):.2f}"
            )

        if (
            args.checkpoint is not None
            and args.checkpoint_interval > 0
            and completed % args.checkpoint_interval == 0
        ):
            save_checkpoint(model, optimizer, completed, args.checkpoint)
            print(f"saved checkpoint to {args.checkpoint}")

    if args.checkpoint is not None:
        save_checkpoint(model, optimizer, args.max_iters, args.checkpoint)
        print(f"saved final checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    train(parse_args())
