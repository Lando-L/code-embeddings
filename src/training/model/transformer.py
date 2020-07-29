import numpy as np
import tensorflow as tf

from training.model.decoder import Decoder
from training.model.encoder import Encoder

"""
Taken from: https://www.tensorflow.org/tutorials/text/transformer
"""

class Transformer(tf.keras.Model):
    def __init__(self,
                 num_paths,
                 num_tokens,
                 num_layers,
                 num_heads,
                 embedding_size,
                 dense_size,
                 path_vocab_size,
                 token_vocab_size,
                 target_vocab_size,
                 maximum_position_encoding,
                 dropout_rate):
        
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_paths,
            num_tokens,
            num_layers,
            num_heads,
            embedding_size,
            dense_size,
            path_vocab_size,
            token_vocab_size,
            dropout_rate
        )

        self.decoder = Decoder(
            num_layers,
            num_heads,
            embedding_size,
            dense_size,
            target_vocab_size,
            maximum_position_encoding,
            dropout_rate
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, X, y, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        path = X['path_input']
        left = X['token_left_input']
        right = X['token_right_input']

        enc_output = self.encoder(path, left, right, training, enc_padding_mask)  # (batch_size, inp_seq_len, embedding_size)

        # dec_output.shape == (batch_size, tar_seq_len, embedding_size)
        dec_output, attention_weights = self.decoder(y, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
