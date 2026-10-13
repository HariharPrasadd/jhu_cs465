#!/usr/bin/env python3
"""Classify text files by comparing two trigram language models."""

import argparse
import logging
import math
from pathlib import Path

import torch

from fileprob import file_log_prob
from probs import LanguageModel


log = logging.getLogger(Path(__file__).stem)


def parse_args() -> argparse.Namespace:
    """Read the two models, first-model prior, and files from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model1", type=Path, help="language model for category 1")
    parser.add_argument("model2", type=Path, help="language model for category 2")
    parser.add_argument(
        "prior",
        type=float,
        help="prior probability of category 1",
    )
    parser.add_argument("test_files", type=Path, nargs="+")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="device to use for PyTorch (cpu, cuda, or mps)",
    )

    # logging goes to stderr so it does not disturb the required stdout format
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not 0.0 <= args.prior <= 1.0:
        parser.error("prior must be between 0 and 1")
    return args


def log_prior(probability: float) -> float:
    """Return log(probability), treating a zero prior as impossible."""
    if probability == 0.0:
        return -math.inf
    return math.log(probability)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # we put the loaded model tensors on the requested device
    torch.set_default_device(args.device)
    model1 = LanguageModel.load(args.model1, device=args.device)
    model2 = LanguageModel.load(args.model2, device=args.device)

    # scores are only comparable when both models use the same finite vocabulary
    if model1.vocab != model2.vocab:
        raise ValueError("The two language models must use the same vocabulary")

    # bayes' theorem lets us ignore the shared p(document) denominator
    # we can rank each class with log p(document | class) + log p(class)
    model1_log_prior = log_prior(args.prior)
    model2_log_prior = log_prior(1.0 - args.prior)
    model1_count = 0
    model2_count = 0

    for test_file in args.test_files:
        model1_score = file_log_prob(test_file, model1) + model1_log_prior
        model2_score = file_log_prob(test_file, model2) + model2_log_prior

        # we give ties to the first model so the result stays deterministic
        if model1_score >= model2_score:
            chosen_model = args.model1
            model1_count += 1
        else:
            chosen_model = args.model2
            model2_count += 1

        print(f"{chosen_model} {test_file}")

    # we print the exact two-line summary format requested in the handout
    total = model1_count + model2_count
    print(
        f"{model1_count} files were more probably from {args.model1} "
        f"({100 * model1_count / total:.2f}%)"
    )
    print(
        f"{model2_count} files were more probably from {args.model2} "
        f"({100 * model2_count / total:.2f}%)"
    )


if __name__ == "__main__":
    main()
