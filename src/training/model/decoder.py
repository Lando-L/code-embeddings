import numpy as np
import tensorflow as tf

from training.model.attention import MultiHeadAttentionLayer
from training.model.feedforward import point_wise_feed_forward_network

"""
Taken from: https://www.tensorflow.org/tutorials/text/transformer
"""

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_heads, dense_size, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttentionLayer(embedding_size, num_heads)
        self.mha2 = MultiHeadAttentionLayer(embedding_size, num_heads)

        self.ffn = point_wise_feed_forward_network(embedding_size, dense_size)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, embedding_size)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, embedding_size)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, embedding_size)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, embedding_size)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, embedding_size)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, embedding_size)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 num_heads,
                 embedding_size,
                 dense_size,
                 target_vocab_size,
                 maximum_position_encoding,
                 dropout_rate):

        super(Decoder, self).__init__()

        self.embedding_size = embedding_size
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, embedding_size)
        self.pos_encoding = self.positional_encoding(maximum_position_encoding)
        
        self.dec_layers = [DecoderLayer(self.embedding_size, num_heads, dense_size, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_angles(self, position, i):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(self.embedding_size))
        return position * angle_rates

    def positional_encoding(self, position):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(self.embedding_size)[np.newaxis, :]
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, embedding_size)
        x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, embedding_size)
        return x, attention_weights
