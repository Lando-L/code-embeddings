from argparse import ArgumentParser
from functools import partial

import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from preprocessing import dataset
from preprocessing import vocabulary
from training import loss, mask, schedule
from training.model import transformer


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def parser():
    """
    Returns the command line args parser.

    Returns
    -------
    parser : ArgumentParser
        The command line args parser.
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument('output')

    arg_parser.add_argument('--dict', required=True)
    arg_parser.add_argument('--data', required=True)

    arg_parser.add_argument('--num-paths', type=int, default=100)
    arg_parser.add_argument('--num-tokens', type=int, default=10)
    arg_parser.add_argument('--num-targets', type=int, default=10)

    arg_parser.add_argument('--num-layers', type=int, default=2)
    arg_parser.add_argument('--num-heads', type=int, default=4)
    arg_parser.add_argument('--embedding-size', type=int, default=32)
    arg_parser.add_argument('--dense-size', type=int, default=64)
    arg_parser.add_argument('--dropout-rate', type=float, default=.2)

    arg_parser.add_argument('--batch-size', type=int, default=128)
    arg_parser.add_argument('--epochs', type=int, default=100)
    arg_parser.add_argument('--seed', type=int, default=239)

    return arg_parser


def evaluate(X, num_targets, tar2idx, model):
    """
    Returns the command line args parser.
    
    Parameters
    ----------
    X : tensor
        The features
    num_targets : int
        The maximum number of targets
    tar2idx : dict
        A dictionary mapping from the target sub-tokens to their indices
    model : tf.keras.models.Model
        The trained model

    Returns
    -------
    y_hat : tensor
        Predictions.
    """
    X = {k: tf.expand_dims(v, 0) for k, v in X.items()}
    y = tf.expand_dims([tar2idx[vocabulary.SOS]], 0)

    for i in range(num_targets):
        enc_padding_mask, combined_mask, dec_padding_mask = mask.create_masks(X['path_input'], y)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = model(X, y, False, enc_padding_mask, combined_mask, dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tar2idx[vocabulary.EOS]:
            return tf.squeeze(y, axis=0)[1:], attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        y = tf.concat([y, predicted_id], axis=-1)

    return tf.squeeze(y, axis=0)[1:], attention_weights


def main():
    args = parser().parse_args()

    # Dictionaries
    print('Loading dictionaries from {}'.format(args.dict))
    subtoken2count, path2count, target2count, max_contexts = vocabulary.load(args.dict)

    idx2sub, sub2idx = vocabulary.to_encoder_decoder(subtoken2count, special_tokens=[vocabulary.PAD, vocabulary.UNK])
    idx2path, path2idx = vocabulary.to_encoder_decoder(path2count, special_tokens=[vocabulary.PAD, vocabulary.UNK])
    idx2tar, tar2idx = vocabulary.to_encoder_decoder(target2count, special_tokens=[vocabulary.PAD, vocabulary.UNK, vocabulary.SOS, vocabulary.EOS])

    token_table = vocabulary.to_table(sub2idx, sub2idx[vocabulary.UNK])
    path_table = vocabulary.to_table(path2idx, path2idx[vocabulary.UNK])
    target_table = vocabulary.to_table(tar2idx, tar2idx[vocabulary.UNK])

    # Dataset
    print('Loading data from {}'.format(args.data))
    dst = dataset.create(
        args.data,
        args.num_paths,
        args.num_tokens,
        args.num_targets,
        token_table,
        path_table,
        target_table
    )

    # Model
    print('Creating model')
    model = transformer.Transformer(
        args.num_paths,
        args.num_tokens,
        args.num_layers,
        args.num_heads,
        args.embedding_size,
        args.dense_size,
        len(idx2path),
        len(idx2sub),
        len(idx2tar),
        1000,
        args.dropout_rate
    )

    # Checkpoints
    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!')

    # Evaluation loop
    print('Setting up evaluation')
    evaluate_fn = partial(evaluate, num_targets=args.num_targets, tar2idx=tar2idx, model=model)

    with open(args.output, 'w') as output:
        for X, y in dst:
            y_hat, weights = evaluate_fn(X)
            y = tf.gather_nd(y, tf.where(y > 3))
            
            real = '_'.join([idx2tar[i] for i in y.numpy()])
            predicted = '_'.join([idx2tar[i] for i in y_hat.numpy()])
        
            output.write(f'Real function name: {real} \n')
            output.write(f'Predicted function name: {predicted} \n')
            output.write('\n')


if __name__ == '__main__':
    main()
