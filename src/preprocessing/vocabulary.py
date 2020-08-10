import pickle
from typing import Dict, List, Tuple

import tensorflow as tf


# Special tokens
SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
UNK = '<UNK>'


def load(path: str) -> Tuple:
    """
    Returns dictionaries of tokens to their number of occurences.

    Parameters
    ----------
    path : str
        The path of the pickled dictionary

    Returns
    -------
    subtoken2count : dict
        A dictionary mapping subtokens to their number of occurences
    node2count : dict
        A dictionary mapping ast path nodes to their number of occurences
    target2count : dict
        A dictionary mapping subtargets to their number of occurences
    max_context : int
        The maximum number of paths per function
    """
    with open(path, 'rb') as file:
        subtoken2count = pickle.load(file)
        node2count = pickle.load(file)
        target2count = pickle.load(file)
        max_contexts = pickle.load(file)
        
        return subtoken2count, node2count, target2count, max_contexts


def to_encoder_decoder(token2count: Dict[str, int], special_tokens: List[str] = [], max_size: int = None) -> Tuple[Dict, Dict]:
    """
    Returns dictionaries of tokens to their number of occurences.

    Parameters
    ----------
    token2count : dict
        A dictionary mapping subtokens to their number of occurences
    special_tokens : list
        A list of special tokens to include
    max_size : int
        The maximum size of the dictionaries

    Returns
    -------
    idx2token : dict
        A dictionary mapping from the indices to their sub-tokens
    token2idx : dict
        A dictionary mapping from the sub-tokens to their indices
    """
    # Initialise dictionaries with special tokens
    offset = len(special_tokens)
    idx2token = dict(enumerate(special_tokens))
    token2idx = dict(zip(idx2token.values(), idx2token.keys()))

    # Sort by most occurences
    sorted_tokens = sorted(token2count, key=token2count.get, reverse=True)[:max_size]

    for idx, token in enumerate(sorted_tokens):
        idx2token[idx + offset] = token
        token2idx[token] = idx + offset

    return idx2token, token2idx


def to_table(token2idx, default_token):
    """
    Returns a tf.lookups for a given subtoken dictionary.

    Parameters
    ----------
    token2idx : dict
        A dictionary mapping from the sub-tokens to their indices
    default_token : int
        The default token used when out of vocabulary tokens are encountered

    Returns
    -------
    table_lookup : tf.lookup
        The table lookup
    """
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(list(token2idx.keys()), list(token2idx.values())),
        default_token
    )
