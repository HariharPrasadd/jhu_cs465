#!/usr/bin/env python3
"""Choose speech-recognition candidates using acoustic and language scores."""

import argparse
import logging
import math
from pathlib import Path
from typing import Iterable, NamedTuple

import torch

from probs import LanguageModel, Wordtype, sequence_trigrams


log = logging.getLogger(Path(__file__).stem)


class Candidate(NamedTuple):
    """Store one candidate transcription and its supplied scores."""

    word_error_rate: float
    acoustic_log_probability: float
    words: tuple[Wordtype, ...]


def parse_args() -> argparse.Namespace:
    """Read the model, utterance files, and device from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="trained language model")
    parser.add_argument("utterance_files", type=Path, nargs="+")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="device used for PyTorch computations",
    )

    # logging stays on stderr so stdout keeps the required format
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )
    return parser.parse_args()


def read_utterance(file: Path) -> tuple[int, list[Candidate]]:
    """Read the reference length and nine candidates from an utterance file."""
    with open(file, "rt") as source:
        first_line = source.readline().split(maxsplit=1)
        if not first_line:
            raise ValueError(f"Empty utterance file: {file}")
        reference_length = int(first_line[0])

        candidates: list[Candidate] = []
        for line_number, line in enumerate(source, start=2):
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"Malformed line {line_number} in {file}")

            word_error_rate = float(fields[0])
            acoustic_log_probability = float(fields[1])
            candidate_length = int(fields[2])
            words = tuple(fields[3:])

            # the supplied length lets us catch a malformed candidate immediately
            if len(words) != candidate_length:
                raise ValueError(
                    f"Expected {candidate_length} words on line {line_number} of {file}, "
                    f"but found {len(words)}"
                )

            candidates.append(
                Candidate(word_error_rate, acoustic_log_probability, words)
            )

    if not candidates:
        raise ValueError(f"No candidate transcriptions in {file}")
    return reference_length, candidates


def language_log_probability(words: Iterable[Wordtype], model: LanguageModel) -> float:
    """Return the base-2 language-model score of one transcription."""
    natural_log_probability = sum(
        model.log_prob(x, y, z)
        for x, y, z in sequence_trigrams(words, model.vocab)
    )

    # acoustic scores use base 2, so the language score must use it too
    return natural_log_probability / math.log(2)


def choose_candidate(candidates: Iterable[Candidate], model: LanguageModel) -> Candidate:
    """Return the candidate with the largest Bayesian score."""
    def score(candidate: Candidate) -> float:
        """Combine the acoustic likelihood and language-model prior."""
        return candidate.acoustic_log_probability + language_log_probability(
            candidate.words, model
        )

    return max(candidates, key=score)


def main() -> None:
    """Choose a transcription for each file and print weighted overall WER."""
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    torch.set_default_device(args.device)
    model = LanguageModel.load(args.model, device=args.device)

    total_errors = 0
    total_reference_words = 0

    for utterance_file in args.utterance_files:
        reference_length, candidates = read_utterance(utterance_file)
        chosen = choose_candidate(candidates, model)

        print(f"{chosen.word_error_rate:.3f} {utterance_file}")
        # the supplied wer is rounded, so recover the integer error count first
        total_errors += round(chosen.word_error_rate * reference_length)
        total_reference_words += reference_length

    overall_error_rate = total_errors / total_reference_words
    print(f"{overall_error_rate:.3f} OVERALL")


if __name__ == "__main__":
    main()
