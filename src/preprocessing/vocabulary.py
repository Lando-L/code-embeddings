import pickle
from typing import Dict, List, Tuple

import tensorflow as tf


SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
UNK = '<UNK>'


def load(path: str) -> Tuple:
    with open(path, 'rb') as file:
        subtoken2count = pickle.load(file)
        node2count = pickle.load(file)
        target2count = pickle.load(file)
        max_contexts = pickle.load(file)
        
        return subtoken2count, node2count, target2count, max_contexts


def to_encoder_decoder(token2count: Dict[str, int], special_tokens: List[str] = [], max_size: int = None) -> Tuple[Dict, Dict]:
    offset = len(special_tokens)
    idx2token = dict(enumerate(special_tokens))
    token2idx = dict(zip(idx2token.values(), idx2token.keys()))

    sorted_tokens = sorted(token2count, key=token2count.get, reverse=True)[:max_size]

    for idx, token in enumerate(sorted_tokens):
        idx2token[idx + offset] = token
        token2idx[token] = idx + offset

    return idx2token, token2idx


def to_table(token2idx, default_token):
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(list(token2idx.keys()), list(token2idx.values())),
        default_token
    )
