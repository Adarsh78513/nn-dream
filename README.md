# nn-dream
### Character-Level Language Model with Tokenization and Transformer Architecture

This repository demonstrates the implementation of a **Character-Level Language Model**(will change soon in future) using **basic tokenization** techniques and **Transformer-based architecture**. The model is designed to predict the next character in a sequence, trained on Shakespeare's works. 

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Tokenization](#tokenization)
- [Model](#model)
- [Training and Generating Text](#training-and-generating-text)
- [Loss Visualization](#loss-visualization)
- [Example Output](#example-output)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository combines two core tasks:
1. **Tokenization**: A basic character-level tokenizer to convert text into integers and back.
2. **Language Modeling**: A Transformer-based model to predict the next character in a sequence.

The model uses **self-attention** and **multi-head attention** mechanisms, focusing on character-level predictions. It learns from Shakespeare's texts and generates text that mimics the style and structure of the original works.

## Features

- Character-level tokenization using a simple dictionary-based encoding and decoding.
- Transformer architecture with self-attention heads and multi-head attention layers.
- Training the model on Shakespeare’s works with a simple PyTorch implementation.
- Ability to generate coherent sequences of text based on an input context.
- Visualization of training and validation loss.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Adarsh78513/nn-dream.git
    cd <repository-directory>
    ```

2. Install required packages:
    ```bash
    pip install torch matplotlib
    ```

3. Download the dataset:
    ```bash
    !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    ```

## Dataset

The model is trained on the **Tiny Shakespeare Dataset**, a plain text file containing all of Shakespeare's works. You can download the dataset with the provided command or link.

## Repository Structure

The repository contains the following key files:

- `tokenization.py`: Implements a simple character-level tokenizer, encoding and decoding text into and from integer representations.
- `language_model.py`: Contains the Transformer-based language model, using tokenized data to predict the next character in a sequence.
- `input.txt`: The dataset file containing Shakespeare's works used for training the model.

## Tokenization

The `tokenizerBPE` file contains the logic for converting characters into integers and back. Tokenization is performed using a Byte Pair Encoding (BPE) approach, which is an advanced subword tokenization technique that balances between character-level and word-level tokenization.
### BPE Tokenizer Implementation

In the `tokenization.py` file, the BPE-based tokenization is implemented as follows:

1. **Character Set Extraction**: Extracts the initial set of unique characters from the dataset.
2. **Pair Merging**: Iteratively merges the most frequent character pairs to form subwords.
3. **Encoding**: Converts text into a list of integer IDs representing subwords.
4. **Decoding**: Converts a list of integer IDs back into human-readable text.

Here’s an example of using the tokenizer:

```python
# Encode a sentence into subwords using BPE
encoded_text = encode_bpe("To be, or not to be.")
print(encoded_text)  # Output: [10, 15, 2, 5, 1, 21, 33, ...]

# Decode the encoded subwords back into text
decoded_text = decode_bpe(encoded_text)
print(decoded_text)  # Output: "To be, or not to be."

Example usage:
```python
encoded_text = encode("To be, or not to be.")
decoded_text = decode(encoded_text)
```

## Model

The `language_model.py` file implements the character-level language model using Transformer-based architecture:
- **Head**: A single self-attention head performing scaled dot-product attention.
- **Multi-Head Attention**: Multiple heads of self-attention in parallel.
- **FeedForward**: A linear layer with ReLU activation and dropout.
- **Transformer Block**: Combines self-attention and feed-forward layers with residual connections.
- **LanguageModel**: The overall architecture including embedding layers, Transformer blocks, and a final layer to predict the next character.

## Training and Generating Text

To train the model, split the dataset into training, validation, and test sets. The model learns to predict the next character based on the sequence of previous characters.

Example training command:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    # Training loop
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

Once trained, the model can generate text from a given context:

```python
context = "To be, or not to be."
generated_text = model.generate(context, max_new_tokens=500)
print(generated_text)
```

## Loss Visualization

The training and validation losses are logged during training and can be visualized using `matplotlib`:

```python
plt.plot(iterations, train_losses, label='Train Loss')
plt.plot(iterations, val_losses, label='Validation Loss')
plt.show()
```

## Example Output

Here is an example of text generated by the model after training:

```text
Context: "Hello"
Generated text: "Hello, fair prince, thou hast found me not..."
```

The results are not of a very high quality because of a small dataset and low pareters. The model does not have enough time to learn what are words, how they correlate etc, but after the training, the generations vaguely feel something like shakesphere but they dont make sense (yet)

## Contributing

Contributions to improve the model's performance and functionality are welcome. You can contribute by:
- Improving the model’s initialization or hyperparameters.
- Adding advanced tokenization (such as subword-level tokenization).
- Improving loss convergence and prediction accuracy.