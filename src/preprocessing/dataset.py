from functools import partial
from typing import Callable, Dict, Generator, Tuple

import numpy as np
import tensorflow as tf

from preprocessing import vocabulary


def __label(text):
    splits = tf.strings.split(tf.strings.strip(text), sep=' ')
    return splits[1:], splits[0][tf.newaxis, ...]


def __split(X, y, num_paths, num_tokens, num_targets, token_table, path_table, target_table):    
    splitted = tf.strings.split(X, sep=',').to_tensor(default_value=vocabulary.PAD, shape=[None, 3])
    
    X_left = tf.strings.split(splitted[:, 0], sep='|').to_tensor(default_value=vocabulary.PAD, shape=[num_paths, num_tokens])
    X_path = tf.strings.split(splitted[:, 1], sep='|').to_tensor(default_value=vocabulary.PAD, shape=[num_paths, num_tokens])
    X_right = tf.strings.split(splitted[:, 2], sep='|').to_tensor(default_value=vocabulary.PAD, shape=[num_paths, num_tokens])
    
    sos = tf.tile([vocabulary.SOS], [y.shape[0]])[..., tf.newaxis]
    eos = tf.tile([vocabulary.EOS], [y.shape[0]])[..., tf.newaxis]
    
    y = tf.concat([sos, tf.strings.split(y, sep='|'), eos], axis=1).to_tensor(default_value=vocabulary.PAD, shape=[None, num_targets])
    
    X = {
        'token_left_input': token_table.lookup(X_left),
        'path_input': path_table.lookup(X_path),
        'token_right_input': token_table.lookup(X_right)
    }

    y = target_table.lookup(tf.reshape(y, [num_targets]))
    
    return X, y


def create(path, num_paths, num_tokens, num_targets, token_table, path_table, target_table) -> tf.data.Dataset:
    return tf.data \
        .TextLineDataset(path) \
        .map(__label) \
        .map(
            partial(
                __split,
                num_paths=num_paths,
                num_tokens=num_tokens,
                num_targets=num_targets,
                token_table=token_table,
                path_table=path_table,
                target_table=target_table
            )
        )
