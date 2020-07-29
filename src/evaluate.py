def evaluate(X):
    X = {k: tf.expand_dims(v, 0) for k,v in X.items()}
    y = tf.expand_dims([tar2idx[vocabulary.SOS]], 0)

    for i in range(10):
        enc_padding_mask, combined_mask, dec_padding_mask = mask.create_masks(X['path_input'], y)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = model(X, 
                                                 y,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tar2idx[vocabulary.EOS]:
            return tf.squeeze(y, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        y = tf.concat([y, predicted_id], axis=-1)

    return tf.squeeze(y, axis=0), attention_weights


# paths

splits = []

with open(args.data, 'r') as file:
    for line in file:
        splits.append(line.split(' '))

# plot
fig = plt.figure(figsize=(16, 8))

attention = tf.squeeze(weights[1]['decoder_layer2_block2'][:, :, :, :13], axis=0).numpy()
token_left = tf.squeeze(features[1]['token_left_input'][:13, :1], axis=1).numpy()
result = results[1][1][1:]

for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head+1)
    ax.matshow(attention[head][:-1, :], cmap='viridis')
    
    fontdict = {'fontsize': 10}
    
    ax.set_xticks(range(len(token_left) + 2))
    ax.set_yticks(range(len(result)))
    ax.set_ylim(len(result) - 0.5, -0.5)
    ax.set_xticklabels(list(range(len(token_left))), fontdict=fontdict, rotation=90)
    ax.set_yticklabels([idx2tar[token] for token in result], fontdict=fontdict)

plt.tight_layout()
plt.show()
