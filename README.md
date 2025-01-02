# LLM-From-Scratch

This project demonstrates the process of building, training, and evaluating a Language Model (LLM) from scratch using PyTorch. The model is based on the GPT architecture and includes advanced techniques like masked multi-head attention, transformer blocks, and post-processing methods such as temperature scaling and top-k sampling.

Table of Contents
Introduction
Features
Requirements
Installation
Usage
Dataset
Model Architecture
Training
Evaluation
Text Generation
Results
Contributing
License
Introduction
This project aims to provide a comprehensive guide to building a Language Model (LLM) from scratch. The model is trained on a small dataset for educational purposes and includes various advanced techniques to improve performance and diversity in text generation.

Features
Data preparation and tokenization
Custom DataLoader for batching and shuffling
GPT-like model architecture with multi-head attention and feed-forward layers
Training loop with loss calculation and optimization
Evaluation metrics including perplexity
Text generation with temperature scaling and top-k sampling
Visualization of training and validation losses
Requirements
Python 3.x
PyTorch
tiktoken
matplotlib
Installation

Usage
Download and prepare the dataset.
Tokenize the dataset and create DataLoader.
Define the model architecture.
Train the model.
Evaluate the model.
Generate text.
Dataset
The dataset used in this project is a small text file named "the-verdict.txt". The text is tokenized using the tiktoken library and split into training and validation sets.

Model Architecture
The model architecture is based on the GPT model and includes the following components:

Token and position embeddings
Multi-head attention
Feed-forward layers
Layer normalization
Dropout
Training
The model is trained using the AdamW optimizer with a learning rate of 0.0004 and weight decay of 0.1. The training loop includes loss calculation, backpropagation, and optimization. Evaluation is performed at regular intervals to monitor training and validation losses.

Evaluation
The model is evaluated using perplexity as an additional metric. The training and validation losses are visualized to understand the training dynamics.

Text Generation
Text generation is performed using techniques like temperature scaling and top-k sampling to increase diversity in the generated text.

Results
The results include the training and validation losses over epochs, perplexity scores, and generated text samples.

Contributing
Contributions are welcome! Please feel free to submit issues and pull requests.
