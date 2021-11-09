# region Imports
import spacy
from spacy.tokens import DocBin
from spacy.tokens.doc import Doc
from pathlib import Path


# endregion

# region Types
Document = Doc


# endregion

# region Functions
def load_the_stranger() -> Document:
    """
    Loads L'étranger by Albert Camus.
    """
    with open(Path(__file__).parent / "l'etranger.bin", 'rb') as file:
        bytes_data = file.read()
    return list(DocBin().from_bytes(bytes_data).get_docs(spacy.load('fr_core_news_md').vocab))[0]


# endregion