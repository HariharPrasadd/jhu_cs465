"""Inspect spaCy dependency parses for the Question 2 comparison."""

import spacy


# Reuse representative sentences from the constituency-parser experiment.
ENGLISH_SENTENCES = [
    "The chef cooked the chicken and served it with rice.",
    "The lawyer interviewed the witness with the camera.",
    "Which report did the senator say the committee had rejected?",
    "The retired police the stations.",
]

# These examples probe Spanish-specific pro-drop, clitics, and word order.
SPANISH_SENTENCES = [
    "Ayer compré un libro interesante.",
    "María se lo dio a Juan.",
    "El libro que me recomendaste fue fascinante.",
    "Llegaron los estudiantes tarde.",
    "A Juan le gustan los libros.",
    "Vi al hombre con el telescopio.",
]


def print_parses(nlp, sentences: list[str]) -> None:
    """Print token-to-head relations for a collection of sentences."""
    for sentence in sentences:
        print(f"## {sentence}")
        print("TOKEN\tDEPENDENCY\tHEAD")
        for token in nlp(sentence):
            print(f"{token.text}\t{token.dep_}\t{token.head.text}")
        print()


def main() -> None:
    """Run comparable English and Spanish dependency-parser experiments."""
    print("# English")
    print_parses(spacy.load("en_core_web_sm"), ENGLISH_SENTENCES)

    print("# Spanish")
    print_parses(spacy.load("es_core_news_sm"), SPANISH_SENTENCES)


if __name__ == "__main__":
    main()
