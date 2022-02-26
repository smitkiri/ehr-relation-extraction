from typing import List, Iterator
import re


def default_tokenizer(sequence: str) -> List[str]:
    """A tokenizer that splits sequence by a whitespace."""
    words = re.split("\n| |\t", sequence)
    tokens = []
    for word in words:
        word = word.strip()

        if not word:
            continue

        tokens.append(word)

    return tokens


def scispacy_plus_tokenizer(sequence: str, scispacy_tok=None) -> Iterator[str]:
    """
    Runs the scispacy tokenizer and removes all tokens with
    just whitespace characters
    """
    if scispacy_tok is None:
        import en_ner_bc5cdr_md
        scispacy_tok = en_ner_bc5cdr_md.load().tokenizer

    scispacy_tokens = list(map(lambda x: str(x), scispacy_tok(sequence)))
    tokens = filter(lambda t: not (' ' in t or '\n' in t or '\t' in t), scispacy_tokens)

    return tokens
