# region Imports
from __future__ import annotations

from typing import List, Union, Optional

import spacy
from spacy.tokens import DocBin
from spacy.tokens.doc import Doc


# endregion

# region Types
Document = Doc
Document_like = Union[str, List[str], Document]


# endregion

# region Functions
def process_document(
    document_like : Document_like,
    nlp : Optional[spacy.lang.fr.French],
) -> Document:
    """
    Turns a (document_like) input into a proper Document.
    """
    nlp = spacy.load('fr_core_news_md') if nlp is None else nlp
    if isinstance(document_like, list) and isinstance(document_like[0], str):
        document_like = ' '.join(document_like)
    if isinstance(document_like, str):
        document_like = nlp(document_like)
    if isinstance(document_like, Document):
        return document_like
    else:
        raise(ValueError(f"Could'nt process this document-like input to a document: {type(document_like)}"))

def save_document(
    document : Document,
    path : str,
) -> None:
    """
    Saves a (document) as a .bin file pointed to by (path).
    """
    assert path.endswith('.bin'), f"Please point to a .bin file, here the file pointed is {path}"
    doc_bin = DocBin(store_user_data=True)
    doc_bin.add(document)
    with open(path, 'wb') as file:
        file.write(doc_bin.to_bytes())


def load_document(
    path : str,
) -> Document:
    """
    Loads a document saved as a .bin file pointed to by (path).
    """
    assert path.endswith('.bin'), f"Please point to a .bin file, here the file pointed is {path}"
    with open(path, 'rb') as file:
        bytes_data = file.read()
    return list(DocBin().from_bytes(bytes_data).get_docs(spacy.load('fr_core_news_md').vocab))[0]


# endregion