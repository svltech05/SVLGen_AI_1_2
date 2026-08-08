import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# Load Real Dataset
# -------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------
# Standardize Features
# -------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# Convert to Torch Tensor
# -------------------------------------------------

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.FloatTensor(y_train).view(-1,1)
y_test = torch.FloatTensor(y_test).view(-1,1)

# -------------------------------------------------
# Define ANN
# -------------------------------------------------

class ANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(30,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x

model = ANN()

# -------------------------------------------------
# Loss & Optimizer
# -------------------------------------------------

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------------------------
# Training
# -------------------------------------------------

epochs = 100

loss_history = []
accuracy_history = []

for epoch in range(epochs):

    # Forward Pass
    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Training Accuracy
    predicted = (outputs >= 0.5).float()
    accuracy = (predicted == y_train).float().mean()

    loss_history.append(loss.item())
    accuracy_history.append(accuracy.item()*100)

    if (epoch+1)%10==0:
        print(f"Epoch {epoch+1:3d}  Loss={loss.item():.4f}  Accuracy={accuracy.item()*100:.2f}%")

# -------------------------------------------------
# Test Accuracy
# -------------------------------------------------

with torch.no_grad():

    outputs = model(X_test)

    predicted = (outputs>=0.5).float()

    test_accuracy = (predicted==y_test).float().mean()

print("\nTest Accuracy = {:.2f}%".format(test_accuracy*100))

# -------------------------------------------------
# Plot Loss Curve
# -------------------------------------------------

plt.figure(figsize=(8,5))
plt.plot(loss_history, linewidth=2)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.grid(True)
plt.show()

# -------------------------------------------------
# Plot Accuracy Curve
# -------------------------------------------------

plt.figure(figsize=(8,5))
plt.plot(accuracy_history, linewidth=2)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()

# -------------------------------------------------
# Predict a New Sample
# -------------------------------------------------

sample = X_test[0].unsqueeze(0)

with torch.no_grad():

    prediction = model(sample)

print("\nPrediction Probability:", prediction.item())

if prediction >= 0.5:
    print("Predicted Class : Benign")
else:
    print("Predicted Class : Malignant")

print("Actual Class :", "Benign" if y_test[0]==1 else "Malignant")