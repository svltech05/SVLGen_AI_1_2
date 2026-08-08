# Simple Neural Network (SNN) in PyTorch -- Explanation

## Architecture

-   Input:3
-   Hidden:5
-   Output:1

Flow: Input -\> Linear(3,5) -\> ReLU -\> Linear(5,1) -\> Output

## Code Explanation

1.  Import torch, nn, optim and matplotlib.
2.  torch.manual_seed(42)` makes random numbers reproducible.
3.  X=torch.randn(20,3)` creates 20 samples with 3 features.
4.  y=torch.randn(20,1)` creates random targets.
5.  nn.Linear(3,5)` creates the hidden layer.
6.  nn.ReLU()` adds non-linearity.
7.  nn.Linear(5,1)` creates the output layer.
8.  forward()` defines data flow.
9.  nn.MSELoss()` computes prediction error.
10. optim.SGD(..., lr=0.01)` updates weights.
11. optimizer.zero_grad()` clears old gradients.
12. loss.backward()` computes gradients.
13. optimizer.step()` updates weights.
14. losses.append(loss.item())` stores loss.
15. plt.plot(losses)` plots the loss curve.

Training cycle:

Random Data -\> Forward Pass -\> Prediction -\> Loss -\> Backpropagation
-\> Weight Update -\> Repeat

epoch :

An epoch is one complete pass of the entire training dataset through the neural network.

In simple words:

Epoch = One full cycle of learning from all the training data.