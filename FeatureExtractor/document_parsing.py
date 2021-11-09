# region Imports
from __future__ import annotations

from typing import Dict, Set, List, Union

from spacy.tokens.doc import Doc
from spacy.tokens.token import Token


# endregion

# region Types
Document = Doc
Document_like = Union[str, List[str], Document]


# endregion

# region Functions
def count_morph_of_type(
        document : Document,
        type : str,
    ) -> Dict[str, int]:
        """
        Counts the number of each appearance of (type) morphology in the (document).
        """
        count = {}
        for token in (token for token in document if token.pos_ == type):
            morph_string = token.morph.__repr__()
            count[morph_string] = count[morph_string] + 1 if morph_string in count else 1
        return count


def count_lemma_of_type(
        document : Document,
        type : str,
    ) -> Dict[str, int]:
        """
        Counts the number of each appearance of (type) lemma in the (document).
        """
        count = {}
        for token in (token for token in document if token.pos_ == type):
            lemma = token.lemma_.lower()
            count[lemma] = count[lemma] + 1 if lemma in count else 1
        return count


def set_from_type(
    document : Document,
    type : str,
) -> Set[Token]:
    """
    Returns the tokens in the (document) that have a certain (type) as a set
    """
    found = set()
    for token in (token for token in document if token.pos_ == type):
        found.add(token)
    return found


def set_from_morph(
    document : Document,
    morph : str,
) -> Set[Token]:
    """
    Returns the tokens in the (document) that have a certain (morph) as a set
    """
    found = set()
    for token in (token for token in document if token.morph.__repr__() == morph):
        found.add(token)
    return found

def find_by_text(
    document : Document,
    text : str,
    return_one : bool = False,
) -> Union[List[Token], Token]:
    """
    Finds the tokens in the (document) where token.text == text and returns them as a list.
    If the (return_one) flag is passed, only returns a single token.
    """
    found = []
    for token in (token for token in document if token.text == text):
        if return_one:
            return token
        else:
            found.append(token)
    return found

def count_ner_by_label(
    document : Document,
    label : str,
) -> Dict[str, int]:
    """
    Given a (document), counts the named entity in this document recognized as (label).
    """
    found = {}
    for token in (token for token in document.ents if token.label_ == label):
        if token.text not in found:
            found[token.text] = 0
        found[token.text] +=1
    return found


# endregion