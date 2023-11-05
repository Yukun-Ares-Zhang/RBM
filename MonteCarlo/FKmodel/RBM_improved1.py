import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import xlwt

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
   def __init__(self,
               U=4, 
               T=0.15,
               n_vis=64,
               n_hin=100):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin, n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.U = U
        self.T = T
        self.n_vis = n_vis
        self.n_hin = n_hin
    
   def sample_from_p(self,p):
       return F.relu(torch.sign(p - (torch.rand(p.size()))))
    
   def v_to_h(self,v):
        p_h = torch.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
   def h_to_v(self,h):
        p_v = torch.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
   def forward(self,v):
        pre_h,h = self.v_to_h(v)
        return h
    
   def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (- hidden_term - vbias_term)
    
rbm = RBM(U, T)
lambd = 1.

#锁定v_bias
rbm.v_bias = nn.Parameter(torch.zeros(rbm.n_vis) + rbm.U/(2*rbm.T))
for name, param in rbm.named_parameters():
    if "v_bias" in name:
         param.requires_grad = False

def train_loop(dataloader, model, loss_fn, penalty_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = - model.free_energy(X.float())
        loss = loss_fn(pred, y)
        l = loss + lambd * penalty_fn(rbm.W)

        # Backpropagation
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if (batch_size * batch) % 10000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
              
def test_loop(dataloader, model, loss_fn, precision):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = - model.free_energy(X.float())
            test_loss += loss_fn(pred, y).item()
            correct += (abs(pred - y) / y < precision).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct
    

def print_test(dataloader, model, loss_fn):    
    f = xlwt.Workbook()
    erg_sheet = f.add_sheet("erg")
    erg_sheet.write(0,0,"test")
    erg_sheet.write(0,1,"pred")
    index = 1
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = - model.free_energy(X.float())
            size = pred.numel()
            assert pred.dim() == 1,"Dimension error."
            for i in range(size):
                erg_sheet.write(index,0,y[i].item())
                erg_sheet.write(index,1,pred[i].item())
                index += 1
    
    f.save("test_result_adam.xls")      
    return correct


def L2_penalty(w):
    assert w.requires_grad == True,\
        "Weights' gradients should be calculated."
    
    return w.pow(2).sum() / 2

learning_rate = 1e-3

def diff_loss(pred: "prediction tensor", label: "label tensor"):
     assert pred.shape==label.shape,\
          "Prediction and Label not in same shape."
     return (pred-label).mean()

loss_fn = nn.L1Loss()
penalty_fn = L2_penalty
optimizer = torch.optim.Adam(rbm.parameters(),lr = learning_rate, betas = (0.9,0.99))
# optimizer = torch.optim.RMSprop(rbm.parameters(),lr = learning_rate, alpha = 0.9)
# optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate, momentum = 0.8)

epochs = 20
PRECISION = 0.05

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, rbm, loss_fn, penalty_fn, optimizer)
    correct = test_loop(test_dataloader, rbm, loss_fn, PRECISION)
    if correct > 0.95:
        # optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate / 10, momentum=0.8)
        # optimizer = torch.optim.RMSprop(rbm.parameters(),lr = learning_rate / 10, alpha = 0.9)
        optimizer = torch.optim.Adam(rbm.parameters(),lr = learning_rate / 10, betas = (0.9,0.99))
    if t == epochs - 1:
        print_test(test_dataloader, rbm, loss_fn)
print("Done!")


print(rbm.W)
torch.save(rbm.W, "./Trained_weight_adam.pth")