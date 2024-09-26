
# Transformer Packages for smaples

This package provides a TensorFlow implementation of a Transformer model for translation tasks. It includes data loading, model definition, and training scripts and docs as well.

## Model Overview

The Transformer model is a neural network architecture introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al. It relies entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution.

### Key Components

1. **Embedding**: Converts input tokens into dense vectors of a fixed size.
2. **Positional Encoding**: Injects positional information into the input embeddings.
3. **Multi-Head Attention**: Computes attention weights and applies them to the input.
4. **Point-Wise Feed Forward Network**: A fully connected feed-forward network applied to each position.
5. **Encoder**: Composed of multiple layers, each containing Multi-Head Attention and Feed Forward Network.
6. **Decoder**: Similar to the Encoder but includes an additional Multi-Head Attention layer to attend to the encoder's output.

### Model Parameters

- `num_layers`: Number of layers in the encoder and decoder.
- `d_model`: Dimension of the embedding space.
- `num_heads`: Number of heads in the Multi-Head Attention mechanism.
- `dff`: Dimension of the feed-forward network.
- `input_vocab_size`: Size of the input vocabulary.
- `target_vocab_size`: Size of the target vocabulary.
- `dropout_rate`: Dropout rate for regularization.

## Data

The dataset used is the `ted_hrlr_translate/pt_to_en` dataset from TensorFlow Datasets. It contains Portuguese to English sentence pairs from TED talks.

### Data Loading and Preprocessing

- **Tokenization**: Sentences are tokenized using the SubwordTextEncoder.
- **Padding**: Sequences are padded to ensure uniform length.
- **Batching**: Data is batched for efficient training.

## Installation

Clone the repository and install the package using pip:

```bash
git clone https://github.com/YShokrollahi/transformers-tf.git
cd transformers-tf
pip install .
```

## Usage

To train the model, run the training script:

```bash
train_transformer
```

### Training

The training script will load the dataset, initialize the Transformer model, and train it for a specified number of epochs. Training progress, including loss and accuracy, will be printed to the console.

## Testing

To run the tests, execute:

```bash
python -m unittest discover tests
```

## Example Usage

You can use the package to translate sentences as follows:

```python
from transformer.train import train_transformer
```
# Train the model
transformer, tokenizer_pt, tokenizer_en = train_transformer(EPOCHS=5)

def evaluate(sentence):
    sentence = tokenizer_pt.encode(sentence)
    sentence = tf.expand_dims([tokenizer_pt.vocab_size] + sentence + [tokenizer_pt.vocab_size + 1], axis=0)

    encoder_input = sentence

    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == tokenizer_en.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def translate(sentence):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])

    return predicted_sentence

# Test the model with some sentences
test_sentences = [
    "este é um problema que temos que resolver.",
    "o mercado de livros é pequeno.",
    "eu gosto de aprender novas línguas."
]

for sentence in test_sentences:
    translated_sentence = translate(sentence)
    print(f'Input: {sentence}')
    print(f'Translation: {translated_sentence}')
\`\`\`
