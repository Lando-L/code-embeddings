from functools import partial
from typing import Callable, Dict, Generator, Tuple

import numpy as np
import tensorflow as tf

from preprocessing import vocabulary


def __label(text):
    """
    Returns the features (function body) and target (function name).

    Parameters
    ----------
    text : str
        A line of training data

    Returns
    -------
    features : tensor
        A tensor of the ast paths
    target : tensor
        A tensor of the function name
    """
    splits = tf.strings.split(tf.strings.strip(text), sep=' ')
    return splits[1:], splits[0][tf.newaxis, ...]


def __split(X, y, num_paths, num_tokens, num_targets, token_table, path_table, target_table):  
    """
    Returns the features and targets split into sub-tokens, sub-paths or sub-targets respectively.

    Parameters
    ----------
    X : tensor
        A tensor of the ast paths
    y : tensor
        A tensor of the function name
    num_paths :  int
        The maximum number of paths
    num_tokens : int
        The maximum number of tokens per path
    num_targets : int
        The maximum number of targets
    token_table : tf.lookup
        A lookup table of tokens
    path_table : tf.lookup
        A lookup table of path nodes
    target_table : tf.lookup
        A lookup table of targets

    Returns
    -------
    features : dict
        A dictionary of the split ast path tensors
    targets : tensor
        A tensor of the split function names
    """
    # Split features into left terminal node, path and right terminal node
    splitted = tf.strings.split(X, sep=',').to_tensor(default_value=vocabulary.PAD, shape=[None, 3])
    
    # Split each sub feature into sub-tokens or sub-paths respectively.
    X_left = tf.strings.split(splitted[:, 0], sep='|').to_tensor(default_value=vocabulary.PAD, shape=[num_paths, num_tokens])
    X_path = tf.strings.split(splitted[:, 1], sep='|').to_tensor(default_value=vocabulary.PAD, shape=[num_paths, num_tokens])
    X_right = tf.strings.split(splitted[:, 2], sep='|').to_tensor(default_value=vocabulary.PAD, shape=[num_paths, num_tokens])
    
    # Create tensors holding <Start Of Sequence> and <End of Sequence> tokens 
    sos = tf.tile([vocabulary.SOS], [y.shape[0]])[..., tf.newaxis]
    eos = tf.tile([vocabulary.EOS], [y.shape[0]])[..., tf.newaxis]
    
    # Add <Start Of Sequence> and <End of Sequence> tokens to the target tensor 
    y = tf.concat([sos, tf.strings.split(y, sep='|'), eos], axis=1).to_tensor(default_value=vocabulary.PAD, shape=[None, num_targets])
    
    # Map tokens to lookup indeces
    X = {
        'token_left_input': token_table.lookup(X_left),
        'path_input': path_table.lookup(X_path),
        'token_right_input': token_table.lookup(X_right)
    }

    y = target_table.lookup(tf.reshape(y, [num_targets]))
    
    return X, y


def create(path, num_paths, num_tokens, num_targets, token_table, path_table, target_table) -> tf.data.Dataset:
    """
    Returns the preprocessed dataset read from a .c2s text document.

    Parameters
    ----------
    path : str
        The path of the .c2s text document.
    num_paths :  int
        The maximum number of paths
    num_tokens : int
        The maximum number of tokens per path
    num_targets : int
        The maximum number of targets
    token_table : tf.lookup
        A lookup table of tokens
    path_table : tf.lookup
        A lookup table of path nodes
    target_table : tf.lookup
        A lookup table of targets

    Returns
    -------
    dataset : tf.data.Dataset
        The preprocessed dataset. 
    """
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
