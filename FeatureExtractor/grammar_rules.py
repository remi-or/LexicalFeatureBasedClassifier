# region Imports
from __future__ import annotations

from typing import Dict, Optional, List
import json
from pathlib import Path

from .resources import UniversalPOStags, InterrogativeWords


# endregion

# region Functions
def count_transitive(
    verb_lemmas : Dict[str, int],
    path : Optional[str] = None,
) -> bool:
    """
    Given a (verb_lemmas) dictionnary containing lemmas and their count, returns the number of transitive verbs.
    This function uses a dictionnary where keys are transitive verbs that is encoded as a json at (json_relative_path).
    """
    if path is None:
        path = Path(__file__).parent / 'data/verbes_transitifs.json'
    with open(path, encoding='utf-8') as file:
        transitives = json.load(file)
    count = 0
    for lemma, cardinal in verb_lemmas.items():
        if lemma in transitives:
            count += cardinal
    return count

def preload_lexique3_freq_table(
    path : Optional[str] = None,
) -> dict:
    """
    Preloads the frequency table pointed to by (path) if path is given.
    """
    if path is None:
        path = Path(__file__).parent / 'data/lemma_to_freq_table.json'
    with open(path, encoding='utf-8') as file:
        lemma_to_freq = json.load(file)
    return lemma_to_freq

def get_lexique_3_subtitles_freqs(
    cgrams : str | List[str],
    lemmas : str | List[str],
    path_or_freq_table : Optional[str | dict] = None,
) -> List[float]:
    """
    Given a list of (lemmas) and their (types), chosen among the universal POS tags, 
    returns their frequencies in the french language according to the lexique.
    A single type / lemma couple can be passed as well, as two strings.
    A single type can be passed for mutliple lemmas as well.
    """
    ## Type checking
    # Single couple case
    if isinstance(cgrams, str) and isinstance(lemmas, str):
        cgrams, lemmas = [cgrams], [lemmas]
    # One type one list case
    elif isinstance(cgrams, str) and isinstance(lemmas, list):
        cgrams = [cgrams] * len(lemmas)
    # Two lists case
    elif isinstance(cgrams, list) and isinstance(lemmas, list):
        pass
    # Error case
    else:
        raise TypeError(f"Unsupported input types: {type(cgrams)}, {type(lemmas)}.\nThis functions excepts str/str, list/list, str/list for types/lemmas")
    
    ## Loading the frequency table
    # Default path is set if None is given
    if path_or_freq_table is None:
        path = Path(__file__).parent / 'data/lemma_to_freq_table.json'
    if isinstance(path_or_freq_table, str):
        # Load the frequency tables
        with open(path, encoding='utf-8') as file:
            lemma_to_freq = json.load(file)
    else:
        lemma_to_freq = path_or_freq_table
    
    ## Main loop
    frequencies = []
    for cgram, lemma in zip(cgrams, lemmas):
        # Check if the type is right
        assert cgram in UniversalPOStags, \
            f"This type isn't an universal POS tag: {cgram}. The universal POS tags are: {UniversalPOStags}"
        # Case where the frequency isn't zero
        try:
            frequencies.append(lemma_to_freq[cgram][lemma])
        # Case where the frequency is zero, excepted exception
        except KeyError:
            frequencies.append(0)
        # Error case, unexcepted exception
        except Exception as unexcepected_exception:
            raise unexcepected_exception
    
    return frequencies


def count_interrogative_words(
    lemmas : Dict[str, int],
    cgram : str,
) -> int:
    """
    Given a (lemmas) dictionnary where lemmas are from (cgram), containing lemmas and their number of appearances in a document,
    counts the number of interrogative words in the said document.
    """
    if cgram not in InterrogativeWords:
        return 0
    else:
        return sum(
            (lemmas[lemma] if lemma in lemmas else 0)
            for lemma in InterrogativeWords[cgram]
        )
    

# endregion