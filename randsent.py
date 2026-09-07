#!/usr/bin/env python3
"""
601.465/665 — Natural Language Processing
Assignment 1: Designing Context-Free Grammars

Assignment written by Jason Eisner
Modified by Kevin Duh
Re-modified by Alexandra DeLucia

Code template written by Alexandra DeLucia,
based on the submitted assignment with Keith Harrigian
and Carlos Aguirre Fall 2019
"""
import argparse
import os
import random
import subprocess
import sys

# Want to know what command-line arguments a program allows?
# Commonly you can ask by passing it the --help option, like this:
#     python randsent.py --help
# This is possible for any program that processes its command-line
# arguments using the argparse module, as we do below.
#
# NOTE: When you use the Python argparse module, parse_args() is the
# traditional name for the function that you create to analyze the
# command line.  Parsing the command line is different from parsing a
# natural-language sentence.  It's easier.  But in both cases,
# "parsing" a string means identifying the elements of the string and
# the roles they play.

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (an argparse.Namespace): Stores command-line attributes
    """
    # Initialize parser
    parser = argparse.ArgumentParser(description="Generate random sentences from a PCFG")
    # Grammar file (required argument)
    parser.add_argument(
        "-g",
        "--grammar",
        type=str, required=True,
        help="Path to grammar file",
    )
    # Start symbol of the grammar
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )
    # Number of sentences
    parser.add_argument(
        "-n",
        "--num_sentences",
        type=int,
        help="Number of sentences to generate (default is 1)",
        default=1,
    )
    # Max number of nonterminals to expand when generating a sentence
    parser.add_argument(
        "-M",
        "--max_expansions",
        type=int,
        help="Max number of nonterminals to expand when generating a sentence",
        default=450,
    )
    # Print the derivation tree for each generated sentence
    parser.add_argument(
        "-t",
        "--tree",
        action="store_true",
        help="Print the derivation tree for each generated sentence",
        default=False,
    )
    return parser.parse_args()


class Grammar:
    def __init__(self, grammar_file):
        """
        Context-Free Grammar (CFG) Sentence Generator

        Args:
            grammar_file (str): Path to a .gr grammar file
        
        Returns:
            self
        """
        # Initialize rules as an empty dict
        self.rules = {}
        self.expansions = 0
        self._load_rules_from_file(grammar_file)

    def _load_rules_from_file(self, grammar_file):
        """
        Read grammar file and store its rules in self.rules

        Args:
            grammar_file (str): Path to the raw grammar file 
        """

        with open(grammar_file, "r") as f:
            # only adds lines that are not commented and not whitespaced to the list
            lines = (line.strip() for line in f if line.strip() and not line.startswith("#"))
                
            for line in lines: 
                # remove inline comments
                if '#' in line:
                    line = line.split('#')[0].strip()

                if not line:
                    continue

                # Check formatting
                if len(line.split("\t")) != 3:
                    raise ValueError("Invalid grammar formatting.")

                # split prob lhs rhs on tabs
                prob, lhs, rhs = line.split("\t")

                # check symbols
                if "(" in lhs or ")" in lhs or "(" in rhs or ")" in rhs:
                    raise ValueError("Invalid grammar symbol.")

                # convert prob to float
                prob = float(prob)

                # check prob
                if prob <= 0:
                    raise ValueError("Rule weight must be positive.")
                
                # split rhs on whitespace
                rhs = rhs.split()
                
                # dictionary format is as follows: {lhs: [list of tuples of (probability, rhs)]}
                if lhs in self.rules:
                    self.rules[lhs].append((prob, rhs))

                else:
                    self.rules[lhs] = [(prob, rhs)]

    def sample(self, derivation_tree=False, max_expansions=450, start_symbol="ROOT", _recursive = False):
        """
        Sample a random sentence from this grammar

        Args:
            derivation_tree (bool): if true, the returned string will represent 
                the tree (using bracket notation) that records how the sentence 
                was derived
                               
            max_expansions (int): max number of nonterminal expansions we allow

            start_symbol (str): start symbol to generate from

        Returns:
            str: the random sentence or its derivation tree
        """
        if not _recursive:
            self.expansions = 0

        if self.expansions >= max_expansions:
            return "..."

        self.expansions += 1

        output = ""

        if(start_symbol in self.rules):
            # taking the different formulations of lhs
            options = self.rules[start_symbol]

            # creating weight and value lists 
            weights = [x[0] for x in options]
            values = [x[1] for x in options]

            # drawing an appropriately weighted formulation of rhs 
            next_symbol = random.choices(values, weights=weights, k=1)[0]

            if(derivation_tree):
                output += f"({start_symbol} "

            for symbol in next_symbol:
                if symbol not in self.rules:
                    output += f"{symbol} "

                else:
                    # set _recursive = True to not reset self.expansions
                    output += self.sample(derivation_tree, max_expansions, symbol, True)

            if(derivation_tree):
                output += ") "

        else:
            raise ValueError("Invalid start symbol.")
        
        return output

####################
### Main Program
####################
def main():
    # Parse command-line options
    # args = parse_args()

    # # Initialize Grammar object
    # grammar = Grammar(args.grammar)

    # # Generate sentences
    # for i in range(args.num_sentences):
    #     # Use Grammar object to generate sentence
    #     sentence = grammar.sample(
    #         derivation_tree=args.tree,
    #         max_expansions=args.max_expansions,
    #         start_symbol=args.start_symbol
    #     )

    #     # Print the sentence with the specified format.
    #     # If it's a tree, we'll pipe the output through the prettyprint script.
    #     if args.tree:
    #         prettyprint_path = os.path.join(os.getcwd(), 'prettyprint')
    #         subprocess.run(
    #             ['perl', prettyprint_path],
    #             input=sentence,
    #             text=True
    #         )
    #     else:
    #         print(sentence)

    grammar = Grammar("grammar.gr")
    sentence = grammar.sample(derivation_tree = True, max_expansions = 5)
    prettyprint_path = os.path.join(os.getcwd(), 'prettyprint')
    subprocess.run(
        ['perl', prettyprint_path],
        input=sentence,
        text=True
    )

if __name__ == "__main__":
    main()
