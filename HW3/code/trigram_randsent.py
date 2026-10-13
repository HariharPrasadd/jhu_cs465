#!/usr/bin/env python3
"""Sample random sentences from a trained trigram language model."""

import argparse
import itertools
import logging
from pathlib import Path

import torch

from probs import LanguageModel


def parse_args() -> argparse.Namespace:
    """Read the model, sample count, and maximum sentence length."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path to the trained model")
    parser.add_argument("num_sentences", type=int, help="number of sentences to sample")
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="maximum number of tokens before truncation (default 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional random seed for reproducible samples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="device to use for PyTorch (cpu, cuda, or mps)",
    )

    args = parser.parse_args()
    if args.num_sentences < 0:
        parser.error("num_sentences must not be negative")
    if args.max_length < 0:
        parser.error("max_length must not be negative")
    return args


def main() -> None:
    args = parse_args()

    # we set the seed only when one is given so normal runs stay random
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # we keep model tensors and new sampling tensors on the same device
    torch.set_default_device(args.device)
    model = LanguageModel.load(args.model, device=args.device)

    for _ in range(args.num_sentences):
        # taking one extra token tells us whether the sentence was truncated
        tokens = list(itertools.islice(model.sample(), args.max_length + 1))
        if len(tokens) > args.max_length:
            tokens = tokens[: args.max_length] + ["..."]
        print(" ".join(tokens))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
