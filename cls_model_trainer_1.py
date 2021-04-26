import numpy
import scipy.special
import torch
import math
from torch import autograd, nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

csv_file_load = pd.read_csv('cls_model_images/poses.csv')
x_csv = csv_file_load.iloc[1:, 1:3]
gamma_pre = numpy.loadtxt("cls_model_images/Classification_Results_Reduced.txt")

x_np = x_csv.values
X = x_np[0:, 0]
Y = x_np[0:, 1]
print(x_np.shape)

# Cuda

def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


# Network architecture

class Model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.Layer1 = nn.Linear(input_size , hidden_size)
        self.Layer2 = nn.Linear(hidden_size, hidden_size)
        self.Layer21 = nn.Linear(hidden_size, hidden_size)
        self.Layer22 = nn.Linear(hidden_size, hidden_size)
        self.Layer3 = nn.Linear(hidden_size, output_size)
        self.Dropout = nn.Dropout(0.5)
        self.Softplus = nn.Softplus()
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.Layer1(x)
        x = F.leaky_relu(x)
        x = self.Layer2(x)
        x = F.leaky_relu(x)
        x = self.Layer21(x)
        x = F.leaky_relu(x)
        x = self.Layer22(x)
        x = F.leaky_relu(x)
        x = self.Layer3(x)
        x = self.Softmax(x)
        return x

# Initialization

input_size = 2
output_size = 10
hidden_size = 50
network = Model(input_size, hidden_size, output_size).cuda()
print("Number of parameters is:", sum(param.numel() for param in network.parameters()))

if torch.cuda.is_available():
    print("Cuda is ACTIVATED")
else:
    print("Cuda is DEACTIVATED")

number_of_epochs = 100
learning_rate = 0.01
crit = nn.MSELoss()
opt = torch.optim.Adam(network.parameters(), lr=learning_rate)

# Network training

network.train()
for epoch in range(number_of_epochs):
#for epoch in range(3):

    x_var = to_var(torch.tensor(x_csv.values).float(), requires_grad=True)
    gamma_var = to_var(torch.from_numpy(gamma_pre).float())

    opt.zero_grad()

    out = network(x_var)
    loss = crit(out, gamma_var)
    print("Loss at epoch ", epoch, " is: ", loss.item())

    loss.backward(out)
    opt.step()

    # test_theta = torch.Tensor([[2, 1, 1], [2, 1, 1]]).requires_grad_()
    # test_gamma = torch.Tensor([[0.5, 0.2, 0.3], [0.5, 0.2, 0.3]])
    # loss_test = crit(test_gamma, test_theta)
    # loss_test.backward(test_theta)

network.eval()
torch.save(network.state_dict(), 'Chair_CLS_Expectation.pkl')
print('Network has been saved')
x_var = to_var(torch.tensor(x_csv.values).float(), requires_grad=True)
out_test = network(x_var)
out_np = out_test.cpu().detach().numpy()
# Change according to network trained

class_index = 0

fig = plt.figure()
ax = fig.gca(projection='3d')
print(type(X))
print(type(Y))
print(type(out_np[0:, class_index]))
ax.plot_trisurf(X, Y, out_np[0:, class_index], cmap=plt.cm.Spectral)
#surf = ax.plot_surface(X, Y, out_np[0:, class_index], rstride=1, cstride=1, cmap=cm.coolwarm,
 #   linewidth=0, antialiased=False)
ax.set_xlabel('$X[cm]$')
ax.set_ylabel('$Y[cm]$')
ax.set_zlabel('Predicted expectation of class probability $\mathbb{P}(c|I)$')
#ax.set_zlabel('Class probability $\mathbb{P}(c_{table}|z)$')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(X, Y, gamma_pre[0:, class_index], cmap=plt.cm.Spectral)
#surf = ax.plot_surface(X, Y, out_np[0:, class_index], rstride=1, cstride=1, cmap=cm.coolwarm,
 #   linewidth=0, antialiased=False)
ax.set_xlabel('$X[cm]$')
ax.set_ylabel('$Y[cm]$')
ax.set_zlabel('Measurements')
#ax.set_zlabel('Class probability $\mathbb{P}(c_{table}|z)$')
plt.show()

