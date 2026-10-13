#!/usr/bin/env python3
"""Run the required experiments and create the graphs for Homework 3, Section 3."""

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fileprob import file_log_prob
from probs import AddLambdaLanguageModel, LanguageModel, Vocab, num_tokens, read_vocab


CODE_DIR = Path(__file__).resolve().parent
HOMEWORK_DIR = CODE_DIR.parent
DATA_DIR = HOMEWORK_DIR / "data" / "gen_spam"
VOCAB_FILE = CODE_DIR / "vocab-genspam.txt"
LAMBDAS = (5.0, 0.5, 0.05, 0.005, 0.0005)
PRIOR_GEN = 0.7


def train_model(vocab: Vocab, corpus: Path, lambda_: float) -> LanguageModel:
    """Train and return one add-lambda language model."""
    model = AddLambdaLanguageModel(vocab, lambda_)
    model.train(corpus)
    return model


def train_pair(vocab: Vocab, suffix: str, lambda_: float) -> tuple[LanguageModel, LanguageModel]:
    """Train matching genuine-email and spam-email language models."""
    train_dir = DATA_DIR / "train"
    gen_model = train_model(vocab, train_dir / f"gen{suffix}", lambda_)
    spam_model = train_model(vocab, train_dir / f"spam{suffix}", lambda_)
    return gen_model, spam_model


def dev_files(category: str | None = None) -> list[Path]:
    """Return the requested development files in a stable order."""
    dev_dir = DATA_DIR / "dev"
    if category is not None:
        return sorted((dev_dir / category).glob("*"))
    return dev_files("gen") + dev_files("spam")


def cross_entropy(model: LanguageModel, files: Iterable[Path]) -> tuple[float, int, float]:
    """Return total bits, token count, and cross-entropy for several files."""
    files = list(files)
    total_log_prob = sum(file_log_prob(file, model) for file in files)
    total_bits = -total_log_prob / math.log(2)
    total_tokens = sum(num_tokens(file) for file in files)
    return total_bits, total_tokens, total_bits / total_tokens


def classify(
    gen_model: LanguageModel,
    spam_model: LanguageModel,
    files: Iterable[Path],
    prior_gen: float,
) -> list[tuple[Path, str, bool]]:
    """Classify files and return each prediction with its correctness."""
    log_prior_gen = -math.inf if prior_gen == 0 else math.log(prior_gen)
    prior_spam = 1.0 - prior_gen
    log_prior_spam = -math.inf if prior_spam == 0 else math.log(prior_spam)
    results = []

    for file in files:
        gen_score = file_log_prob(file, gen_model) + log_prior_gen
        spam_score = file_log_prob(file, spam_model) + log_prior_spam
        predicted = "gen" if gen_score >= spam_score else "spam"
        actual = file.parent.name
        results.append((file, predicted, predicted == actual))

    return results


def error_rate(results: Iterable[tuple[Path, str, bool]]) -> float:
    """Return the fraction of classifications that were incorrect."""
    results = list(results)
    errors = sum(not correct for _, _, correct in results)
    return errors / len(results)


def log_sigmoid(value: float) -> float:
    """Compute the natural log of a sigmoid without numerical underflow."""
    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def all_spam_prior_cutoff(
    gen_model: LanguageModel,
    spam_model: LanguageModel,
    files: Iterable[Path],
) -> tuple[float, Path]:
    """Find the log-prior boundary below which every file is classified as spam."""
    cutoffs = []
    for file in files:
        gen_log_prob = file_log_prob(file, gen_model)
        spam_log_prob = file_log_prob(file, spam_model)

        # keeping the cutoff in log space avoids rounding tiny priors down to zero
        log_cutoff = log_sigmoid(spam_log_prob - gen_log_prob)
        cutoffs.append((log_cutoff, file))

    # every file is spam only below the earliest gen decision boundary
    return min(cutoffs, key=lambda item: item[0])


def file_length(file: Path) -> int:
    """Read the document length encoded in a development filename."""
    return int(file.name.split(".")[1])


def length_bin(length: int) -> str:
    """Place a document length into one of the graph's fixed-width bins."""
    if length <= 25:
        return "1-25"
    if length <= 50:
        return "26-50"
    if length <= 100:
        return "51-100"
    if length <= 200:
        return "101-200"
    if length <= 500:
        return "201-500"
    return "501+"


def graph_length_results(results: list[tuple[Path, str, bool]]) -> list[tuple[str, int, float]]:
    """Graph classification error rate by document-length bin."""
    labels = ["1-25", "26-50", "51-100", "101-200", "201-500", "501+"]
    totals = {label: 0 for label in labels}
    errors = {label: 0 for label in labels}

    for file, _, correct in results:
        label = length_bin(file_length(file))
        totals[label] += 1
        errors[label] += int(not correct)

    rows = [(label, totals[label], errors[label] / totals[label]) for label in labels]
    percentages = [100 * rate for _, _, rate in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, percentages, color="#4361a3")
    ax.set_xlabel("Document length in words")
    ax.set_ylabel("Classification error rate (%)")
    ax.set_title("Add-lambda classifier performance by document length")
    ax.set_ylim(0, max(percentages) * 1.25)
    ax.grid(axis="y", alpha=0.25)

    # the sample counts make the reliability of each bin easier to judge
    for bar, (_, count, rate) in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            100 * rate + 0.25,
            f"{100 * rate:.1f}%\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(HOMEWORK_DIR / "section3_length_error.png", dpi=180)
    plt.close(fig)
    return rows


def graph_learning_curve(rows: list[tuple[int, float]]) -> None:
    """Graph development error rate against relative training-set size."""
    sizes = [size for size, _ in rows]
    percentages = [100 * rate for _, rate in rows]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(sizes, percentages, marker="o", color="#9b3f42", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [f"{size}x" for size in sizes])
    ax.set_xlabel("Relative training-set size")
    ax.set_ylabel("Development error rate (%)")
    ax.set_title("Learning curve for the add-lambda classifier")
    ax.grid(alpha=0.25)
    ax.set_ylim(min(percentages) - 0.5, max(percentages) + 0.8)

    for size, percentage in zip(sizes, percentages):
        ax.annotate(
            f"{percentage:.2f}%",
            (size, percentage),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(HOMEWORK_DIR / "section3_learning_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run every required Section 3 experiment and print its measurements."""
    vocab = read_vocab(VOCAB_FILE)
    all_dev = dev_files()
    gen_dev = dev_files("gen")
    spam_dev = dev_files("spam")

    # part a uses the add-one models from section 2
    add_one_gen, add_one_spam = train_pair(vocab, "", 1.0)
    add_one_results = classify(add_one_gen, add_one_spam, all_dev, PRIOR_GEN)
    print(f"3(a) error rate: {100 * error_rate(add_one_results):.5f}%")

    # part c finds the exact decision boundary instead of guessing priors
    log_cutoff, cutoff_file = all_spam_prior_cutoff(add_one_gen, add_one_spam, all_dev)
    print(f"3(c) log10 all-spam prior cutoff: {log_cutoff / math.log(10):.5f}")
    print(f"3(c) limiting file: {cutoff_file}")

    models_by_lambda = {}
    entropy_rows = []
    for lambda_ in LAMBDAS:
        gen_model, spam_model = train_pair(vocab, "", lambda_)
        models_by_lambda[lambda_] = (gen_model, spam_model)

        gen_bits, gen_tokens, gen_entropy = cross_entropy(gen_model, gen_dev)
        spam_bits, spam_tokens, spam_entropy = cross_entropy(spam_model, spam_dev)
        combined_entropy = (gen_bits + spam_bits) / (gen_tokens + spam_tokens)
        entropy_rows.append((lambda_, gen_entropy, spam_entropy, combined_entropy))
        print(
            f"lambda={lambda_:g} gen={gen_entropy:.5f} "
            f"spam={spam_entropy:.5f} combined={combined_entropy:.5f}"
        )

    best_gen = min(entropy_rows, key=lambda row: row[1])
    best_spam = min(entropy_rows, key=lambda row: row[2])
    best_combined = min(entropy_rows, key=lambda row: row[3])
    lambda_star = best_combined[0]
    print(f"3(d) best gen: lambda={best_gen[0]:g}, entropy={best_gen[1]:.5f}")
    print(f"3(d) best spam: lambda={best_spam[0]:g}, entropy={best_spam[2]:.5f}")
    print(f"3(e) lambda*: {lambda_star:g}")
    print(f"3(e) combined entropy: {best_combined[3]:.5f}")

    # part f uses the selected shared lambda on the same development files
    best_gen_model, best_spam_model = models_by_lambda[lambda_star]
    best_results = classify(best_gen_model, best_spam_model, all_dev, PRIOR_GEN)
    length_rows = graph_length_results(best_results)
    for label, count, rate in length_rows:
        print(f"3(f) {label}: n={count}, error={100 * rate:.5f}%")

    # part h keeps the vocabulary and lambda fixed while training data grows
    learning_rows = []
    for size, suffix in ((1, ""), (2, "-times2"), (4, "-times4"), (8, "-times8")):
        gen_model, spam_model = train_pair(vocab, suffix, lambda_star)
        results = classify(gen_model, spam_model, all_dev, PRIOR_GEN)
        rate = error_rate(results)
        learning_rows.append((size, rate))
        print(f"3(h) {size}x training data: error={100 * rate:.5f}%")

    graph_learning_curve(learning_rows)


if __name__ == "__main__":
    main()
