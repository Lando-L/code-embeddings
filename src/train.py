from argparse import ArgumentParser
from functools import partial
import math
import time

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
    arg_parser.add_argument('--dict', required=True)
    arg_parser.add_argument('--train', required=True)
    arg_parser.add_argument('--test', required=True)

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


def train_step(X, y, model, optimizer, train_loss, train_accuracy):
    """
    Trains the model for one batch.

    Parameters
    ----------
    X : tensor
        The features
    y : tensor
        The targets
    model : tf.model.Model
        The model
    optimizer : tf.keras.optimizers
        The optimizer
    train_loss : tf.Variable
        The variable keeping track of the training loss
    train_accuracy : tf.Variable
        The variable keeping track of the training accuracy
    """
    y_inp = y[:, :-1]
    y_real = y[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = mask.create_masks(X['path_input'], y_inp)

    with tf.GradientTape() as tape:
        predictions, _ = model(X, y_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        _loss = loss.loss_function(y_real, predictions)

    gradients = tape.gradient(_loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(_loss)
    train_accuracy(y_real, predictions)


def test_step(X, y, model, test_loss, test_accuracy):
    """
    Evaluates the model for one batch.

    Parameters
    ----------
    X : tensor
        The features
    y : tensor
        The targets
    model : tf.model.Model
        The model
    test_loss : tf.Variable
        The variable keeping track of the testing loss
    test_accuracy : tf.Variable
        The variable keeping track of the testing accuracy
    """
    y_inp = y[:, :-1]
    y_real = y[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = mask.create_masks(X['path_input'], y_inp)
    predictions, _ = model(X, y_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
    _loss = loss.loss_function(y_real, predictions)

    test_loss(_loss)
    test_accuracy(y_real, predictions)


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
    print('Loading train data from {}'.format(args.train))
    train = dataset.create(
        args.train,
        args.num_paths,
        args.num_tokens,
        args.num_targets,
        token_table,
        path_table,
        target_table
    ).shuffle(10000, seed=args.seed, reshuffle_each_iteration=True).batch(args.batch_size)

    test = dataset.create(
        args.test,
        args.num_paths,
        args.num_tokens,
        args.num_targets,
        token_table,
        path_table,
        target_table
    ).batch(args.batch_size)

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

    # Hyperparameters
    print('Setting up training')
    learning_rate = schedule.CustomSchedule(args.embedding_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Checkpoints
    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!')

    # Logs
    train_log_dir = './logs/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    test_log_dir = './logs/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Train Loop
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    signature = [
        {'token_left_input': tf.TensorSpec(shape=(None, args.num_paths, args.num_tokens), dtype=tf.int32),
         'token_right_input': tf.TensorSpec(shape=(None, args.num_paths, args.num_tokens), dtype=tf.int32),
         'path_input': tf.TensorSpec(shape=(None, args.num_paths, args.num_tokens), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(None, args.num_targets), dtype=tf.int32)
    ]

    train_on_batch = tf.function(
        partial(train_step, model=model, optimizer=optimizer, train_loss=train_loss, train_accuracy=train_accuracy),
        input_signature=signature
    )

    test_on_batch = tf.function(
        partial(test_step, model=model, test_loss=test_loss, test_accuracy=test_accuracy),
        input_signature=signature
    )

    print('Training for {} epochs'.format(args.epochs))
    for epoch in range(args.epochs):
        start = time.time()
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for (X_train, y_train) in train:
            train_on_batch(X_train, y_train)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        for (X_test, y_test) in test:
            test_on_batch(X_test, y_test)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        if (epoch + 1) % 10 == 0:
            ckpt_save_path = ckpt_manager.save()

            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        print(
            'Epoch {} Loss {:.4f} Accuracy {:.4f} Validation Loss {:.4f} Validation Accuracy {:.4f}'.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result(),
                test_loss.result(),
                test_accuracy.result()
            )
        )

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == '__main__':
    main()
