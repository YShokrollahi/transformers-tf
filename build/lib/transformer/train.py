import tensorflow as tf
from transformer.model import Transformer
from transformer.data_loader import load_dataset
from transformer.layers import positional_encoding

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

def train_transformer(EPOCHS=5):
    train_dataset, val_dataset, tokenizer_pt, tokenizer_en = load_dataset()

    num_layers = 4
    d_model = 512
    dff = 2048
    num_heads = 8
    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers, d_model, num_heads, dff,
        input_vocab_size, target_vocab_size,
        pe_input=input_vocab_size,
        pe_target=target_vocab_size,
        rate=dropout_rate)

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function(experimental_relax_shapes=True)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    for epoch in range(EPOCHS):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            if batch % 5 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    return transformer, tokenizer_pt, tokenizer_en
