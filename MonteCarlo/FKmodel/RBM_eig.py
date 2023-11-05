import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import bitflip_class as bf 
import random

U = 4.
T = 0.15
Nx = 8
Ny = Nx
N = Nx * Ny
h_num = 100
DATA_NUM = 50000

training_data = torch.load("./Traning_data_eig.pth")
test_data = torch.load("./Test_data_eig.pth")

batch_size = 5
train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

class EigTrainer(nn.Module):
    def __init__(self, U=4, T=0.15, v_num=N):
        super(EigTrainer, self).__init__()
        self.single_layer_mapping = nn.Sequential(
            nn.Linear(v_num, v_num),
        )
        self.U = U
        self.T = T
    
    def forward(self, x):
        eigenValue = self.single_layer_mapping(x.to(torch.float))
        return eigenValue

eigen_trainer = EigTrainer(U, T)
lambd = 1.
def train_loop(dataloader, model, loss_fn, penalty_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, eig, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model.forward(X)
        weight, bias = list(model.parameters())
        loss = loss_fn(pred, eig)
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
        for X, eig, y in dataloader:
            pred = model.forward(X)
            test_loss += loss_fn(pred, eig).item()
            correct += ((abs(pred - eig) / eig < precision).type(torch.float).sum(dim=1) / N).mean()
            # 每个batch中 每个pred（1*64向量）误差小于PRECISION的entry个数/64 的平均值

    test_loss /= num_batches
    correct /= num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct
    

def L2_penalty(w):
    assert w.requires_grad == True,\
        "Weights' gradients should be calculated."
    
    return w.pow(2).sum() / 2

learning_rate = 1e-3
loss_fn = nn.L1Loss()
penalty_fn = L2_penalty
optimizer = torch.optim.SGD(eigen_trainer.parameters(), lr=learning_rate)

epochs = 20
PRECISION = 0.05

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, eigen_trainer, loss_fn, penalty_fn, optimizer)
    correct = test_loop(test_dataloader, eigen_trainer, loss_fn, PRECISION)
    if correct > 0.95:
        optimizer = torch.optim.SGD(eigen_trainer.parameters(), lr=learning_rate / 10)
print("Done!")

weight, bias = list(eigen_trainer.parameters())
print(weight)
# torch.save(weight, "./Trained_weight.pth")
# for i in range(10):
#     X, y = next(iter(test_dataloader))
#     pred = eigen_trainer.forward(X)
#     loss = loss_fn(pred, eig).item()
#     print(pred)
#     print(y)
#     print(f"loss = {loss}")  

  