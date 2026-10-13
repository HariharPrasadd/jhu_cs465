#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separating business logic from other stuff.  (9/19/2019)

# Patched by Arya McCarthy <arya@jhu.edu> to fix a counting issue that
# evidently was known pre-2016 but then stopped being handled?

# Further refactoring by Jason Eisner <jason@cs.jhu.edu> 
# and Brian Lu <zlu39@jhu.edu>.  (9/26/2021)

from __future__ import annotations

import logging
import math
import pickle
import sys

from pathlib import Path

import torch
from torch import nn
from torch import optim
from jaxtyping import Float
from typeguard import typechecked
from typing import Counter, Collection
from collections import Counter

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union

Wordtype = str  # if you decide to integerize the word types, then change this to int
Vocab    = Collection[Wordtype]   # and change this to Integerizer[str]
Zerogram = Tuple[()]
Unigram  = Tuple[Wordtype]
Bigram   = Tuple[Wordtype, Wordtype]
Trigram  = Tuple[Wordtype, Wordtype, Wordtype]
Ngram    = Union[Zerogram, Unigram, Bigram, Trigram]
Vector   = List[float]
TorchScalar = Float[torch.Tensor, ""] # a torch.Tensor with no dimensions, i.e., a scalar


##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words


##### UTILITY FUNCTIONS FOR CORPUS TOKENIZATION

def read_tokens(file: Path, vocab: Optional[Vocab] = None) -> Iterable[Wordtype]:
    """Iterator over the tokens in file.  Tokens are whitespace-delimited.
    If vocab is given, then tokens that are not in vocab are replaced with OOV."""

    # OPTIONAL SPEEDUP: You may want to modify this to integerize the
    # tokens, using integerizer.py as in previous homeworks.
    # In that case, redefine `Wordtype` from `str` to `int`.

    # PYTHON NOTE: This function uses `yield` to return the tokens one at
    # a time, rather than constructing the whole sequence and using
    # `return` to return it.
    #
    # A function that uses `yield` is called a "generator."  As with other
    # iterators, it computes new values only as needed.  The sequence is
    # never fully constructed as an single object in memory.
    #
    # You can iterate over the yielded sequence, for example, like this:
    #      for token in read_tokens(my_file, vocab):
    #          process(token)
    # Whenever the `for` loop needs another token, read_tokens magically picks up 
    # where it left off and continues running until the next `yield` statement.

    with open(file) as f:
        for line in f:
            for token in line.split():
                if vocab is None or token in vocab:
                    yield token
                else:
                    yield OOV  # replace this out-of-vocabulary word with OOV
            yield EOS  # Every line in the file implicitly ends with EOS.


def num_tokens(file: Path) -> int:
    """Give the number of tokens in file, including EOS."""
    return sum(1 for _ in read_tokens(file))


def read_trigrams(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file.  Each triple (x,y,z) is a token z
    (possibly EOS) with a left context (x,y)."""
    x, y = BOS, BOS
    for z in read_tokens(file, vocab):
        yield (x, y, z)
        if z == EOS:
            x, y = BOS, BOS  # reset for the next sequence in the file (if any)
        else:
            x, y = y, z  # shift over by one position.


def sequence_trigrams(tokens: Iterable[Wordtype], vocab: Vocab) -> Iterable[Trigram]:
    """Yield trigrams for one token sequence, including its final EOS event."""
    x, y = BOS, BOS

    for token in tokens:
        # candidate words use the same fixed vocabulary as ordinary corpus text
        z = token if token in vocab else OOV
        yield x, y, z
        x, y = y, z

    yield x, y, EOS


def draw_trigrams_forever(file: Path, 
                          vocab: Vocab, 
                          randomize: bool = False) -> Iterable[Trigram]:
    """Infinite iterator over trigrams drawn from file.  We iterate over
    all the trigrams, then do it again ad infinitum.  This is useful for 
    SGD training.  
    
    If randomize is True, then randomize the order of the trigrams each time.  
    This is more in the spirit of SGD, but the randomness makes the code harder to debug, 
    and forces us to keep all the trigrams in memory at once.
    """
    trigrams = read_trigrams(file, vocab)
    if not randomize:
        import itertools
        return itertools.cycle(trigrams)  # repeat forever
    else:
        import random
        pool = tuple(trigrams)   
        while True:
            for trigram in random.sample(pool, len(pool)):
                yield trigram

##### READ IN A VOCABULARY (e.g., from a file created by build_vocab.py)

def read_vocab(vocab_file: Path) -> Vocab:
    vocab: Vocab = set()
    with open(vocab_file, "rt") as f:
        for line in f:
            word = line.strip()
            vocab.add(word)
    log.info(f"Read vocab of size {len(vocab)} from {vocab_file}")
    # Convert from an unordered Set to an ordered List.  This ensures that iterating
    # over the vocab will always hit the words in the same order, so that you can 
    # safely store a list or tensor of embeddings in that order, for example.
    return sorted(vocab)   
    # Alternatively, you could choose to represent a Vocab as an Integerizer (see above).
    # Then you won't need to sort, since Integerizers already have a stable iteration order.

##### LANGUAGE MODEL PARENT CLASS

class LanguageModel:

    def __init__(self, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.progress = 0   # To print progress.

        self.event_count:   Counter[Ngram] = Counter()  # numerator c(...) function.
        self.context_count: Counter[Ngram] = Counter()  # denominator c(...) function.
        # In this program, the argument to the counter should be an Ngram, 
        # which is always a tuple of Wordtypes, never a single Wordtype:
        # Zerogram: context_count[()]
        # Bigram:   context_count[(x,y)]   or equivalently context_count[x,y]
        # Unigram:  context_count[(y,)]    or equivalently context_count[y,]
        # but not:  context_count[(y)]     or equivalently context_count[y]  
        #             which incorrectly looks up a Wordtype instead of a 1-tuple

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    # We need to collect two kinds of n-gram counts.
    # To compute p(z | xy) for a trigram xyz, we need c(xy) for the 
    # denominator and c(yz) for the backed-off numerator.  Both of these 
    # look like bigram counts ... but they are not quite the same thing!
    #
    # For a sentence of length N, we are iterating over trigrams xyz where
    # the position of z falls in 1 ... N+1 (so z can be EOS but not BOS),
    # and therefore
    # the position of y falls in 0 ... N   (so y can be BOS but not EOS).
    # 
    # When we write c(yz), we are counting *events z* with *context* y:
    #         c(yz) = |{i in [1, N]: w[i-1] w[i] = yz}|
    # We keep these "event counts" in `event_count` and use them in the numerator.
    # Notice that z=BOS is not possible (BOS is not a possible event).
    # 
    # When we write c(xy), we are counting *all events* with *context* xy:
    #         c(xy) = |{i in [1, N]: w[i-2] w[i-1] = xy}|
    # We keep these "context counts" in `context_count` and use them in the denominator.
    # Notice that y=EOS is not possible (EOS cannot appear in the context).
    #
    # In short, c(xy) and c(yz) count the training bigrams slightly differently.  
    # Likewise, c(y) and c(z) count the training unigrams slightly differently.
    #
    # Note: For bigrams and unigrams that don't include BOS or EOS -- which
    # is most of them! -- `event_count` and `context_count` will give the
    # same value.  So you could save about half the memory if you were
    # careful to store those cases only once.  (How?)  That would make the
    # code slightly more complicated, but would be worth it in a real system.

    def count_trigram_events(self, trigram: Trigram) -> None:
        """Record one token of the trigram and also of its suffixes (for backoff)."""
        (x, y, z) = trigram
        self.event_count[(x, y, z )] += 1
        self.event_count[   (y, z )] += 1
        self.event_count[      (z,)] += 1  # the comma is necessary to make this a tuple
        self.event_count[        ()] += 1

    def count_trigram_contexts(self, trigram: Trigram) -> None:
        """Record one token of the trigram's CONTEXT portion, 
        and also the suffixes of that context (for backoff)."""
        (x, y, _) = trigram    # we don't care about z
        self.context_count[(x, y )] += 1
        self.context_count[   (y,)] += 1
        self.context_count[     ()] += 1

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes an estimate of the trigram log probability log p(z | x,y)
        according to the language model.  The log_prob is what we need to compute
        cross-entropy and to train the model.  It is also unlikely to underflow,
        in contrast to prob.  In many models, we can compute the log_prob directly, 
        rather than first computing the prob and then calling math.log."""
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling log_prob on an instance of LanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.log_prob is not implemented yet (you should override LanguageModel.log_prob)"
        )

    def sample(self) -> Iterable[Wordtype]:
        """Yield a random token sequence from the language model."""
        vocab = tuple(self.vocab)
        x, y = BOS, BOS

        while True:
            # softmax turns the model's log probabilities into sampling weights
            log_probs = torch.tensor(
                [self.log_prob(x, y, z) for z in vocab],
                dtype=torch.float64,
            )
            probabilities = torch.softmax(log_probs, dim=0)

            # multinomial rolls one weighted die over the whole vocabulary
            index = torch.multinomial(probabilities, num_samples=1).item()
            z = vocab[index]

            # eos ends the sequence but is not part of the returned sentence
            if z == EOS:
                return

            yield z
            x, y = y, z

    def save(self, model_path: Path) -> None:
        log.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            # torch.save is similar to pickle.dump but handles tensors too
        log.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = 'cpu') -> "LanguageModel":
        log.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
            # torch.load is similar to pickle.load but handles tensors too
            # map_location allows loading tensors on different device than saved
            # weights_only=False loads the whole object including method code 
            #    (that's a security risk, so you'd better fully trust the file at model_path!).
        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls} but got {type(model)} from file {model_path}")
        log.info(f"Loaded model from {model_path}")
        return model

    def train(self, file: Path) -> None:
        """Create vocabulary and store n-gram counts.  In subclasses, we might
        override this with a method that computes parameters instead of counts."""

        log.info(f"Training from corpus {file}")

        # Clear out any previous training.
        self.event_count   = Counter()
        self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
            self.count_trigram_contexts(trigram)
            self.show_progress()

        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.event_count[()]} tokens")

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")


##### SPECIFIC FAMILIES OF LANGUAGE MODELS

class CountBasedLanguageModel(LanguageModel):

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # For count-based language models, it is usually convenient
        # to compute the probability first (by dividing counts) and
        # then taking the log.
        prob = self.prob(x, y, z)
        if prob == 0.0:
            return -math.inf
        return math.log(prob)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == CountBasedLanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling prob on an instance of CountBasedLanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.prob is not implemented yet (you should override CountBasedLanguageModel.prob)"
        )

class UniformLanguageModel(CountBasedLanguageModel):
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(CountBasedLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab)
        if lambda_ < 0.0:
            raise ValueError(f"Negative lambda argument of {lambda_} could result in negative smoothed probs")
        self.lambda_ = lambda_

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        return ((self.event_count[x, y, z] + self.lambda_) /
                (self.context_count[x, y] + self.lambda_ * self.vocab_size))

        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.


class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Compute an add-lambda trigram probability with recursive backoff."""

        def backed_off_prob(context: Ngram) -> float:
            """Compute the probability of z after a context of up to two words."""
            # the empty context backs off one final time to the uniform model
            if len(context) == 0:
                lower_order_prob = 1 / self.vocab_size
            else:
                lower_order_prob = backed_off_prob(context[1:])

            # adding z creates the event whose count belongs in the numerator
            event = context + (z,)
            assert self.event_count[event] <= self.context_count[context]

            return (
                self.event_count[event]
                + self.lambda_ * self.vocab_size * lower_order_prob
            ) / (self.context_count[context] + self.lambda_ * self.vocab_size)

        return backed_off_prob((x, y))


class EmbeddingLogLinearLanguageModel(LanguageModel, nn.Module):
    # this class is both a language model and a pytorch module

    def __init__(
        self,
        vocab: Vocab,
        lexicon_file: Path,
        l2: float,
        epochs: int,
        batch_size: int = 1,
    ) -> None:
        super().__init__(vocab)
        if l2 < 0:
            raise ValueError(f"Negative regularization strength {l2}")
        if epochs < 0:
            raise ValueError(f"Negative number of epochs {epochs}")
        if batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        self.l2: float = l2
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.eta0: float = 0.01 if lexicon_file.name.startswith("chars-") else 0.00001

        lexicon, self.dim = self._read_lexicon(lexicon_file)
        if OOL not in lexicon:
            raise ValueError(f"Lexicon {lexicon_file} does not contain {OOL}")

        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}

        # contexts can contain bos even though bos is not a predicted vocabulary item
        context_words = [BOS] + list(self.vocab)
        self.context_to_index = {
            word: index for index, word in enumerate(context_words)
        }

        # buffers move with the model but are not changed by the optimizer
        self.register_buffer(
            "context_embeddings",
            torch.stack([self._lookup_embedding(word, lexicon) for word in context_words]),
        )
        self.register_buffer(
            "vocab_embeddings",
            torch.stack([self._lookup_embedding(word, lexicon) for word in self.vocab]),
        )

        # parameters are the matrices that sgd learns
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)

    @staticmethod
    def _read_lexicon(lexicon_file: Path) -> tuple[dict[Wordtype, torch.Tensor], int]:
        """Read word embeddings and return them with their dimensionality."""
        lexicon: dict[Wordtype, torch.Tensor] = {}

        with open(lexicon_file, "rt") as source:
            header = source.readline().split()
            if len(header) != 2:
                raise ValueError(f"Malformed lexicon header in {lexicon_file}")
            _, dim_text = header
            dim = int(dim_text)

            for line_number, line in enumerate(source, start=2):
                fields = line.split()
                if not fields:
                    continue
                word, *values = fields
                if len(values) != dim:
                    raise ValueError(
                        f"Expected {dim} values on line {line_number} of {lexicon_file}"
                    )
                lexicon[word] = torch.tensor([float(value) for value in values])

        return lexicon, dim

    @staticmethod
    def _lookup_embedding(
        word: Wordtype, lexicon: dict[Wordtype, torch.Tensor]
    ) -> torch.Tensor:
        """Look up a word vector, backing off to OOL when necessary."""
        # oov represents a whole class, so it must always use the ool vector
        if word == OOV:
            return lexicon[OOL]

        # lowercasing also supports the lowercase words-gs lexicons
        return lexicon.get(word, lexicon.get(word.lower(), lexicon[OOL]))

    def _embedding(self, word: Wordtype) -> torch.Tensor:
        """Return the stored context embedding for a word."""
        index = self.context_to_index.get(word, self.context_to_index[OOV])
        return self.context_embeddings[index]

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Return log p(z | xy) according to this language model."""
        return self.log_prob_tensor(x, y, z).item()

    @typechecked
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TorchScalar:
        """Return the same value as log_prob, but stored as a tensor."""
        
        z_index = self.word_to_index.get(z, self.word_to_index[OOV])

        # log softmax computes the normalizer stably over the whole vocabulary
        return torch.log_softmax(self.logits(x, y), dim=0)[z_index]

    def logits(self, x: Wordtype, y: Wordtype) -> Float[torch.Tensor,"vocab"]:
        """Return a vector of the logs of the unnormalized probabilities f(xyz) * θ
        for the various types z in the vocabulary.
        These are commonly known as "logits" or "log-odds": the values that you
        exponentiate and renormalize in order to get a probability distribution."""
        # each row in vocab_embeddings supplies one possible z vector
        context_score = self._embedding(x) @ self.X + self._embedding(y) @ self.Y
        return context_score @ self.vocab_embeddings.T

    def batch_logits(self, trigrams: Sequence[Trigram]) -> torch.Tensor:
        """Return one vector of logits for every trigram context in a batch."""
        x_embeddings = torch.stack([self._embedding(x) for x, _, _ in trigrams])
        y_embeddings = torch.stack([self._embedding(y) for _, y, _ in trigrams])

        # matrix multiplication evaluates every context and candidate z together
        context_scores = x_embeddings @ self.X + y_embeddings @ self.Y
        return context_scores @ self.vocab_embeddings.T

    def log_prob_batch(self, trigrams: Sequence[Trigram]) -> torch.Tensor:
        """Return the selected log probability for every trigram in a batch."""
        if not trigrams:
            return torch.empty(0)

        z_indices = torch.tensor(
            [self.word_to_index.get(z, self.word_to_index[OOV]) for _, _, z in trigrams],
            dtype=torch.long,
        )
        log_probabilities = torch.log_softmax(self.batch_logits(trigrams), dim=1)
        rows = torch.arange(len(trigrams))
        return log_probabilities[rows, z_indices]

    def regularizer(self, token_count: int) -> TorchScalar:
        """Return the per-example L2 penalty from the training objective."""
        squared_weights = sum(parameter.square().sum() for parameter in self.parameters())
        return (self.l2 / token_count) * squared_weights

    def make_optimizer(self) -> optim.Optimizer:
        """Construct the optimizer used for parameter updates."""
        return optim.SGD(self.parameters(), lr=self.eta0)

    def train(self, file: Path):    # type: ignore
        """Train the model parameters with stochastic gradient descent."""
        log.info(f"Training from corpus {file}")

        # zero initialization makes the untrained model uniform
        for parameter in self.parameters():
            nn.init.zeros_(parameter)

        token_count = num_tokens(file)
        trigrams = list(read_trigrams(file, self.vocab))
        optimizer = self.make_optimizer()

        log.info(f"Start optimizing on {token_count} training tokens...")
        for epoch in range(1, self.epochs + 1):
            epoch_objective = 0.0

            for start in range(0, token_count, self.batch_size):
                batch = trigrams[start : start + self.batch_size]
                optimizer.zero_grad()

                # minimizing the negative objective is gradient ascent on f
                objective = (
                    self.log_prob_batch(batch).sum()
                    - len(batch) * self.regularizer(token_count)
                )
                (-objective).backward()
                optimizer.step()

                # each example should have equal weight in the epoch average
                epoch_objective += objective.item()

            print(f"epoch {epoch}: F = {epoch_objective / token_count}")

        log.info(f"Finished training on {token_count} tokens")


class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    def __init__(
        self,
        vocab: Vocab,
        lexicon_file: Path,
        l2: float,
        epochs: int,
        batch_size: int = 1,
    ) -> None:
        super().__init__(vocab, lexicon_file, l2, epochs, batch_size)

        # beta learns how strongly to trust the unigram count feature
        self.beta = nn.Parameter(torch.zeros(()), requires_grad=True)
        self.unigram_weights = nn.Parameter(
            torch.zeros(self.vocab_size), requires_grad=True
        )
        self.bigram_weights = nn.Parameter(torch.empty(0), requires_grad=True)
        self.register_buffer("unigram_features", torch.zeros(self.vocab_size))
        self.bigram_feature_indices: dict[
            Wordtype, tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _bigram_bias(self, y: Wordtype) -> torch.Tensor:
        """Return learned indicator-feature weights for bigrams beginning with y."""
        bias = torch.zeros(self.vocab_size)
        indices = self.bigram_feature_indices.get(y)
        if indices is not None:
            z_indices, parameter_indices = indices
            bias = bias.scatter(0, z_indices, self.bigram_weights[parameter_indices])
        return bias

    def logits(self, x: Wordtype, y: Wordtype) -> Float[torch.Tensor, "vocab"]:
        """Return base logits augmented with learned unigram frequencies."""
        return (
            super().logits(x, y)
            + self.beta * self.unigram_features
            + self.unigram_weights
            + self._bigram_bias(y)
        )

    def batch_logits(self, trigrams: Sequence[Trigram]) -> torch.Tensor:
        """Return augmented logits for every context in a batch."""
        bigram_biases = torch.stack(
            [self._bigram_bias(y) for _, y, _ in trigrams]
        )
        return (
            super().batch_logits(trigrams)
            + self.beta * self.unigram_features
            + self.unigram_weights
            + bigram_biases
        )

    def make_optimizer(self) -> optim.Optimizer:
        """Construct SGD with a larger step for the new indicator features."""
        # dense embedding features have larger values and need the smaller rate
        return optim.SGD(
            [
                {"params": [self.X, self.Y, self.beta], "lr": self.eta0},
                {
                    "params": [self.unigram_weights, self.bigram_weights],
                    "lr": 0.001,
                },
            ]
        )

    def train(self, file: Path):    # type: ignore
        """Collect unigram features and train all parameters together."""
        trigrams = list(read_trigrams(file, self.vocab))
        counts: Counter[Wordtype] = Counter(z for _, _, z in trigrams)
        bigram_counts: Counter[Bigram] = Counter((y, z) for _, y, z in trigrams)

        # log counts keep very frequent words from dominating the feature scale
        features = torch.tensor(
            [math.log(counts[word] + 1) for word in self.vocab]
        )
        self.unigram_features.copy_(features)

        # each frequent training bigram gets its own learned indicator weight
        features_by_context: dict[Wordtype, list[tuple[int, int]]] = {}
        parameter_count = 0
        for (y, z), count in sorted(bigram_counts.items()):
            if count < 3:
                continue
            z_index = self.word_to_index[z]
            features_by_context.setdefault(y, []).append(
                (z_index, parameter_count)
            )
            parameter_count += 1

        self.bigram_weights = nn.Parameter(
            torch.zeros(parameter_count), requires_grad=True
        )
        self.bigram_feature_indices = {
            y: (
                torch.tensor([z_index for z_index, _ in pairs], dtype=torch.long),
                torch.tensor(
                    [parameter_index for _, parameter_index in pairs],
                    dtype=torch.long,
                ),
            )
            for y, pairs in features_by_context.items()
        }
        super().train(file)
