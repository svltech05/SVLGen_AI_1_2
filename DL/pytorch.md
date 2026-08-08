Pytorch
---------------------------------------------------------------------
PyTorch is an open-source Machine Learning (ML) and Deep Learning (DL) framework developed by the Meta AI Research team. It is widely used for building and training neural networks because it is simple, flexible, and Python-friendly.

Features of PyTorch :
----------------------------
Open-source and free to use
Easy to learn and implement
Uses Python programming language
Supports Dynamic Computational Graphs
GPU acceleration using CUDA
Rich library for Deep Learning
Large community support
Excellent debugging capabilities

Why Use PyTorch?

PyTorch is popular because it allows developers to:

Build neural networks easily
Train AI models efficiently
Perform automatic differentiation
Execute code dynamically
Deploy models in production

Applications of PyTorch :

PyTorch is used in many AI applications such as:

Image Classification
Object Detection
Face Recognition
Natural Language Processing (NLP)
Speech Recognition
Chatbots
Recommendation Systems
Medical Image Analysis
Self-driving Cars
Generative AI (LLMs)

-------------------------------------------------------------
Components of PyTorch :

1. Tensor

A Tensor is the basic data structure in PyTorch.

It is similar to a NumPy array but can run on both:

CPU
GPU

2. Autograd

Autograd automatically computes gradients during backpropagation.

3. Neural Network Module (torch.nn)

The torch.nn module provides ready-made layers for building neural networks.

Example layers:

Linear Layer
Convolution Layer
Dropout
Batch Normalization
Activation Functions

4. Optimizers (torch.optim)

Optimizers update model parameters during training.

Common optimizers:

SGD
Adam
RMSprop
AdamW

5. Loss Functions

Loss functions measure how well the model is performing.

Common loss functions:

MSELoss
CrossEntropyLoss
BCELoss
L1Loss

6. DataLoader

DataLoader helps load data efficiently.

Features:

Batch processing
Data shuffling
Parallel loading

Advantages of PyTorch :
Easy syntax
Pythonic coding style
Dynamic computation graph
Faster prototyping
GPU support
Excellent for research
Strong community support

Pytorch work flow :
--------------------
Dataset
    ↓
DataLoader
    ↓
Neural Network
    ↓
Forward Pass
    ↓
Loss Calculation
    ↓
Back propagation
    ↓
Optimizer
    ↓
Update Weights
    ↓
Repeat (Epochs)