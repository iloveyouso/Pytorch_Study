import torch
import numpy as np
import matplotlib.pyplot as plt

# Generate 100 random data points
x_train = torch.rand(100, 1) * 10
y_train = 3 * x_train + 2 + torch.randn(100, 1)  # y = 3x + 2 + noise

W = torch.zeros(1,requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W,b], lr=0.01)

nb_epochs = 1000

for epoch in range(1,nb_epochs+1):
    hypothesis = x_train *  W + b
    cost = torch.mean((hypothesis - y_train)**2)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
print(f"y = {round(W.item(),2)}x + {round(b.item(),2)}")

# Plotting
plt.scatter(x_train, y_train, color='blue')  # plot the data points
x = np.linspace(0, 10, 100)
y = W.item() * x + b.item()
plt.plot(x, y, '-r', label=f'y={round(W.item(),2)}x+{round(b.item(),2)}')
plt.title('Graph of the regression line')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='best')
plt.grid()
plt.savefig('DeepLearningForAll/1. Linear Regression/plot.png')