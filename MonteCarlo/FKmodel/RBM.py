import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import bitflip_class as bf 
import random
import bitflip

U = 4.
T = 0.15
Nx = 8
Ny = Nx
N = Nx * Ny
h_num = 100
DATA_NUM = 50000

training_data = torch.load("./Traning_data.pth")
test_data = torch.load("./Test_data.pth")

batch_size = 5
train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

class RBM(nn.Module):
    def __init__(self, U=4, T=0.15, v_num=N, h_num=h_num):
        super(RBM, self).__init__()
        self.single_layer_mapping = nn.Linear(v_num, h_num)
        self.U = U
        self.T = T
    
    def forward(self, x):
        minus_F = self.single_layer_mapping(x.to(torch.float)).exp().add(1).log().sum(dim=1)
        minus_F += x.sum(dim=1) * self.U / (2*self.T)
        return minus_F

rbm = RBM(U, T)
lambd = 1.
def train_loop(dataloader, model, loss_fn, penalty_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model.forward(X)
        weight, bias = list(model.parameters())
        loss = loss_fn(pred, y)
        l = loss + lambd * penalty_fn(weight)

        # Backpropagation
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if batch % 5000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
              
def test_loop(dataloader, model, loss_fn, precision):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model.forward(X)
            test_loss += loss_fn(pred, y).item()
            correct += (abs(pred - y) / y < precision).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct
    

def L2_penalty(w):
    assert w.requires_grad == True,\
        "Weights' gradients should be calculated."
    
    return w.pow(2).sum() / 2

learning_rate = 1e-3
loss_fn = nn.L1Loss()
penalty_fn = L2_penalty
optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate)

epochs = 10
PRECISION = 0.05

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, rbm, loss_fn, penalty_fn, optimizer)
    correct = test_loop(test_dataloader, rbm, loss_fn, PRECISION)
    if correct > 0.95:
        optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate / 10)
print("Done!")

weight, bias = list(rbm.parameters())
print(weight)
torch.save(weight, "./Trained_weight.pth")
# for i in range(10):
#     X, y = next(iter(test_dataloader))
#     pred = rbm.forward(X)
#     loss = loss_fn(pred, y).item()
#     print(pred)
#     print(y)
#     print(f"loss = {loss}")  

  