import torch
import numpy as np
import torch.nn.functional as F

class MultiVariableLinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train = torch.FloatTensor([[73., 80., 75.],
                            [93., 88., 93.],
                            [89., 91., 90.],
                            [96., 98., 100.],
                            [73., 66., 70.]]).to(device)
y_train = torch.FloatTensor([[152.], [185.], [180.], [196.], [142.]]).to(device)

# model initialization = hypothesis
model = MultiVariableLinearRegressionModel().to(device)

# optimizer
optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)

nb_epochs = 20000
for epoch in range(nb_epochs + 1):
    hypothesis = model(x_train)
    cost = F.mse_loss(hypothesis,y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(epoch,nb_epochs,hypothesis.squeeze().detach(),cost.item()))
    


