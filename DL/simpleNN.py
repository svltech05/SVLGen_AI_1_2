import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)
X=torch.randn(20,3)
y=torch.randn(20,1)

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(3,5)
        self.relu=nn.ReLU()
        self.output=nn.Linear(5,1)
    
    def forward(self,x):
        x=self.relu(self.hidden(x))
        return self.output(x)

model=SimpleNN()
criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

losses=[]
for epoch in range(100):
    pred=model(X)
    loss=criterion(pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1)%10==0:
        print(f'Epoch {epoch+1}: {loss.item():.4f}')

print('\nPredictions:')
print(model(X))

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()