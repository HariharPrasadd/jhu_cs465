"""Run the Berkeley Neural Parser on sentences selected for Question 1."""

from pathlib import Path

import benepar


# Keep the downloaded model inside HW4 so the experiment remains reproducible.
MODEL_PATH = Path(__file__).parent / "models" / "benepar_en3"

# Tokens are supplied explicitly to avoid depending on a separate NLTK tokenizer.
SENTENCES = [
    ("PP attachment", "The lawyer interviewed the witness with the camera ."),
    (
        "Nested relative clauses",
        "The book that the professor who I met recommended was fascinating .",
    ),
    (
        "Long-distance question",
        "Which report did the senator say the committee had rejected ?",
    ),
    ("Coordination", "The chef cooked the chicken and served it with rice ."),
    ("Garden path: old", "The old man the boats ."),
    ("Designed adversarial sentence", "The retired police the stations ."),
    (
        "Garden path: complex",
        "The complex houses married and single soldiers and their families .",
    ),
    (
        "Garden path: cotton",
        "The cotton clothing is made of grows in Mississippi .",
    ),
    ("Reduced relative", "The horse raced past the barn fell ."),
]


def main() -> None:
    """Load the parser once and print one labeled tree per test sentence."""
    parser = benepar.Parser(str(MODEL_PATH))
    for label, sentence in SENTENCES:
        print(f"## {label}\n{sentence}")
        print(parser.parse(sentence.split()))
        print()


if __name__ == "__main__":
    main()
