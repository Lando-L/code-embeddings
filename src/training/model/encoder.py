import tensorflow as tf

from training.model.attention import MultiHeadAttentionLayer
from training.model.feedforward import point_wise_feed_forward_network

"""
Taken from: https://www.tensorflow.org/tutorials/text/transformer
"""

class TokenEncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TokenEncoderLayer, self).__init__()
        
    def call(self, x):
        return tf.reduce_mean(x, axis=-2) # batch_size, num_paths, embeddings_size


class PathEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_paths, num_tokens):
        super(PathEncoderLayer, self).__init__()
        
        self.embedding_size = embedding_size
        self.num_paths = num_paths
        self.num_tokens = num_tokens
        self.encoder = tf.keras.layers.GRU(self.embedding_size, return_state=True)
        
    def call(self, x):
        flatten = tf.reshape(x, [-1, self.num_tokens, self.embedding_size]) # batch_size * num_paths, num_tokens, embedding_size
        _, state = self.encoder(flatten) # batch_size * num_paths, embedding_size
        reshaped = tf.reshape(state, [-1, self.num_paths, self.embedding_size]) # batch_size, num_paths, embedding_size
        
        return reshaped


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_heads, dense_size, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttentionLayer(embedding_size, num_heads)
        self.ffn = point_wise_feed_forward_network(embedding_size, dense_size)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, embedding_size)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, embedding_size)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, embedding_size)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, embedding_size)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_paths,
                 num_tokens,
                 num_layers,
                 num_heads,
                 embedding_size,
                 dense_size,
                 path_vocab_size,
                 token_vocab_size,
                 dropout_rate):
        
        super(Encoder, self).__init__()

        self.emb_path = tf.keras.layers.Embedding(path_vocab_size, embedding_size)
        self.emb_token = tf.keras.layers.Embedding(token_vocab_size, embedding_size)

        self.enc_path = PathEncoderLayer(embedding_size, num_paths, num_tokens)
        self.enc_token = TokenEncoderLayer()

        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense = tf.keras.layers.Dense(embedding_size, activation='tanh')

        self.enc_layers = [EncoderLayer(embedding_size, num_heads, dense_size, dropout_rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, path, left, right, training, mask):
        path = self.emb_path(path)
        path = self.enc_path(path)
        
        left = self.emb_token(left)
        left = self.enc_token(left)

        right = self.emb_token(right)
        right = self.enc_token(right)

        x = self.concat([path, left, right])
        x = self.dense(x)
        x = self.dropout(x, training=training)
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training, mask)
        
        return x  # (batch_size, input_seq_len, embedding_size)
