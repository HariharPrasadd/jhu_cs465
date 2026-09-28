#!/usr/bin/env python3
"""
Determine most similar words in terms of their word embeddings.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from integerize import Integerizer   # look at integerize.py for more info

# Needed for Python's optional type annotations.
# We've included type annotations and recommend that you do the same, 
# so that mypy (or a similar package) can catch type errors in your code.
from typing import List, Optional

try:
    # PyTorch is your friend. Not using it will make your program so slow.
    # And it's also required for this assignment. ;-)
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    import torch
    import torch.nn as nn
except ImportError:
    print("\nERROR! You need to install Miniconda, then create and activate the nlp-class environment.  See the INSTRUCTIONS file.\n")
    raise


log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.

# Logging is in general a good practice to monitor the behavior of your code
# while it's running. Compared to calling `print`, it provides two benefits.
# 
# - It prints to standard error (stderr), not standard output (stdout) by
#   default.  So these messages will normally go to your screen, even if
#   you have redirected stdout to a file.  And they will not be seen by
#   the autograder, so the autograder won't be confused by them.
# 
# - You can configure how much logging information is provided, by
#   controlling the logging 'level'. You have a few options, like
#   'debug', 'info', 'warning', and 'error'. By setting a global flag,
#   you can ensure that the information you want - and only that info -
#   is printed. As an example:
#        >>> try:
#        ...     rare_word = "prestidigitation"
#        ...     vocab.get_counts(rare_word)
#        ... except KeyError:
#        ...     log.error(f"Word that broke the program: {rare_word}")
#        ...     log.error(f"Current contents of vocab: {vocab.data}")
#        ...     raise  # Crash the program; can't recover.
#        >>> log.info(f"Size of vocabulary is {len(vocab)}")
#        >>> if len(vocab) == 0:
#        ...     log.warning(f"Empty vocab. This may cause problems.")
#        >>> log.debug(f"The values are {vocab}")
#   If we set the log level to be 'INFO', only the log.info, log.warning,
#   and log.error statements will be printed. You can calibrate exactly how 
#   much info you need, and when. None of these pollute stdout with things 
#   that aren't the real 'output' of your program.
# 
# In `parse_args`, we provided two command-line options to control the logging level.
# The default level is 'INFO'. You can lower it to 'DEBUG' if you pass '--verbose'
# and you can raise it to 'WARNING' if you pass '--quiet'.
#
# More info: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
# 
# In all the starter code for the NLP course, we've elected to create a separate
# logger for each source code file, stored in a variable named log that
# is globally visible throughout the file.  That way, calls like log.info(...)
# will use the logger for the current source code file and thus their output will 
# helpfully show the filename.  You could configure the current file's logger using
# log.basicConfig(...), whereas logging.basicConfig(...) affects all of the loggers.
# The command-line options affect all of the loggers.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings", type=Path, help="path to word embeddings file")
    parser.add_argument("word", type=str, help="word to look up")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error(f"Embeddings file {args.embeddings} not found")
    if (args.minus is None) != (args.plus is None):  # != is the XOR operation!
        parser.error("Must include both `--plus` and `--minus` or neither")

    return args

class Lexicon:
    """
    Class that manages a lexicon and can compute similarity.

    >>> my_lexicon = Lexicon.from_file(my_file)
    >>> my_lexicon.find_similar_words("bagpipe")
    """

    def __init__(self) -> None:
        """Initialize empty vocab and embeddings objects"""
        self.vocab: Integerizer[str] = Integerizer()
        self.embeddings: torch.Tensor = torch.empty(
            (0, 0), dtype=torch.float32
        )

    @classmethod
    def from_file(cls, file: Path) -> Lexicon:
        """Load words and their embedding vectors from a text file."""

        words: List[str] = [] # list for words
        vectors: List[List[float]] = [] # list for corresponding vectors

        with open(file) as f:
            expected_words, dimension = map(int, next(f).split()) # get number of words and dimension of vector from header

            for line_number, line in enumerate(f, start=2):
                fields = line.split() # get words and vectors

                if not fields: 
                    continue # skip empty lines

                # get word and vector
                word = fields[0]
                vector = [float(value) for value in fields[1:]]

                # confirm vector dimension matches specification
                if len(vector) != dimension:
                    raise ValueError(
                        f"Line {line_number}: expected {dimension} values, "
                        f"but found {len(vector)}"
                    )

                # add word and vector to list
                words.append(word)
                vectors.append(vector)

        # check that number of words matches specificaiton
        if len(words) != expected_words:
            raise ValueError(
                f"Header says {expected_words} words, but found {len(words)}"
            )

        lexicon = cls()
        lexicon.vocab = Integerizer(words)
        lexicon.embeddings = torch.tensor(vectors, dtype=torch.float32)
        return lexicon
            

    def find_similar_words(
        self, word: str, *, plus: Optional[str] = None, minus: Optional[str] = None
    ) -> List[str]:
        """Find most similar words, in terms of embeddings, to a query."""
        if (minus is None) != (plus is None):  # != is the XOR operation!
            raise TypeError("Must include both of `plus` and `minus` or neither.")

        # Find the embedding for the main query word.
        word_index = self.vocab.index(word)
        if word_index is None:
            raise KeyError(f"Word not found: {word}")

        query = self.embeddings[word_index].clone()

        # For analogies, calculate: word - minus + plus.
        minus_index = None
        plus_index = None

        if minus is not None and plus is not None:
            minus_index = self.vocab.index(minus)
            plus_index = self.vocab.index(plus)

            if minus_index is None:
                raise KeyError(f"Word not found: {minus}")
            if plus_index is None:
                raise KeyError(f"Word not found: {plus}")

            query = (
                query
                - self.embeddings[minus_index]
                + self.embeddings[plus_index]
            )

        # Compare the query with every embedding simultaneously.
        similarities = nn.functional.cosine_similarity(
            self.embeddings,
            query.unsqueeze(0), # unsqueeze to (1, 50) so you can broadcast against the full embedding matrix
            dim=1,
        )

        # Prevent the input words themselves from appearing in the results.
        similarities[word_index] = float("-inf")

        if minus_index is not None and plus_index is not None:
            similarities[minus_index] = float("-inf")
            similarities[plus_index] = float("-inf")

        # Get the indices of the 10 highest cosine similarities.
        best_indices = torch.topk(similarities, k=10).indices.tolist()

        # Translate the integer indices back into words.
        return [self.vocab[index] for index in best_indices]


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    lexicon = Lexicon.from_file(args.embeddings)
    similar_words = lexicon.find_similar_words(
        args.word, plus=args.plus, minus=args.minus
    )
    print(" ".join(similar_words))  # print all words on one line, separated by spaces


if __name__ == "__main__":
    main()
