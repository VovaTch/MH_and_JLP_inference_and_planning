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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

cpv_size = 10
input_size = 2

csv_file_load = pd.read_csv('cls_model_images/poses.csv')
x_csv = csv_file_load.iloc[1:, 1:3]
gamma_pre = numpy.loadtxt("cls_model_images/Classification_Results_Reduced.txt")

x_np = x_csv.values
X = x_np[0:, 0]
Y = x_np[0:, 1]

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

class Model_Cov(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, cpv_size=cpv_size):
        super(Model_Cov, self).__init__()
        self.Layer1 = nn.Linear(input_size , hidden_size)
        self.Layer2 = nn.Linear(hidden_size, hidden_size)
        self.Layer21 = nn.Linear(hidden_size, hidden_size)
        self.Layer22 = nn.Linear(hidden_size, hidden_size)
        self.Layer3 = nn.Linear(hidden_size, output_size)
        self.Dropout = nn.Dropout(0.5)
        self.Softplus = nn.Softplus()
        self.Softmax = nn.Softmax(dim=1)
        self.cpv_size = cpv_size

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
        x_size = x.size()

        if x.dim() is not 1:
            x_size = x_size[0]

            for idx in range(x_size):

                if torch.cuda.is_available():
                    r_matrix = self.r_matrix_output_conversion(x[idx]).cuda()
                else:
                    r_matrix = self.r_matrix_output_conversion(x[idx])

                if idx is 0:
                    x_matrix = r_matrix
                    x_matrix = torch.unsqueeze(x_matrix, dim=2)
                else:
                    r_matrix = torch.unsqueeze(r_matrix, dim=2)
                    x_matrix = torch.cat((x_matrix, r_matrix), dim=2)

        else:

            if torch.cuda.is_available():
                x_matrix = self.r_matrix_output_conversion(x).cuda()
            else:
                x_matrix = self.r_matrix_output_conversion(x)

        return x_matrix

    def r_matrix_output_conversion(self, input):
        r_matrix = torch.zeros([self.cpv_size, self.cpv_size])
        input_index = 0
        for idx_x in range(self.cpv_size):
            for idx_y in range(self.cpv_size - idx_x):
                r_matrix[idx_x][self.cpv_size - idx_y - 1]= input[input_index]
                if idx_x is self.cpv_size - idx_y - 1 and input[input_index] <= 0:
                    r_matrix[idx_x][self.cpv_size - idx_y - 1] = - r_matrix[idx_x][self.cpv_size - idx_y - 1]
                input_index = input_index + 1

        return r_matrix

# Covariance loss

class Cov_Loss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gamma, exp_cpv, r_matrix):

        ctx.save_for_backward(gamma, exp_cpv, r_matrix)
        log_det = 0
        l2_norm = 0

        r_size = r_matrix.size()
        if r_matrix.dim() is 3:
            r_size = r_size[2]

            for idx in range(r_size):
                log_det += -2 * torch.logdet(r_matrix[:, :, idx])
                adding = torch.add(gamma[idx, :], -exp_cpv[idx, :])
                l2_norm += torch.norm(torch.mv(r_matrix[:, :, idx], adding))

        else:

            log_det += -2 * torch.logdet(r_matrix)
            l2_norm += torch.norm(torch.mm(r_matrix, torch.add(gamma, -exp_cpv)))

        return log_det + l2_norm
        #return to_var(log_det + l2_norm, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):

        gamma, exp_cpv, r_matrix = ctx.saved_tensors
        if torch.cuda.is_available():
            r_matrix_der = torch.zeros(r_matrix.size()).cuda()
        else:
            r_matrix_der = torch.zeros(r_matrix.size())

        gamma_der = None
        exp_cpv_der = None
        # r_flattened_der = None
        # r_matrix_der = None

        r_size = r_matrix.size()
        r_inner_size = r_size[0]

        flatten_size = int(r_inner_size ** 2 / 2 + r_inner_size / 2)

        # Gradient for predictions and classifier outputs, probably not needed, will return if it is needed
        # if ctx.needs_input_grad[0]:


        # Gradient for root information matrix
        if ctx.needs_input_grad[2]:

            if r_matrix.dim() is 3:

                r_size = r_size[2]
                r_flattened_der = torch.zeros([flatten_size, r_size])


                for idx in range(r_size):
                    for idx_r in range(r_inner_size):
                        r_matrix_der[idx_r, idx_r, idx] = -2 / r_matrix[idx_r, idx_r, idx]

                    dif_vector = torch.add(gamma[idx, :], -exp_cpv[idx, :])
                    outer_product = torch.ger(dif_vector, dif_vector)
                    dif_matrix = 2 * torch.matmul(r_matrix[:, :, idx], outer_product)
                    r_matrix_der[:, :, idx] += torch.add(r_matrix_der[:, :, idx], dif_matrix)

                    # index = 0
                    # for idx_x in range(r_inner_size):
                    #     for idx_y in range(r_inner_size - idx_x):
                    #         r_flattened_der[index, idx] = r_matrix_der[idx_x, r_inner_size - idx_y - 1, idx]
                    #         index += 1

        print(gamma_der)
        print(exp_cpv_der)
        print(r_matrix_der)
        return gamma_der, exp_cpv_der, r_matrix_der
    # @staticmethod
    # def to_var(x, requires_grad=False):
    #     if torch.cuda.is_available():
    #         x = x.cuda()
    #     return Variable(x, requires_grad=requires_grad)


class MyLoss(nn.Module):
    def forward(self, gamma, exp_cpv, r_matrix):
        return Cov_Loss.apply(gamma, exp_cpv, r_matrix)




# Initialize networks

exp_net = Model(input_size, 50, cpv_size).cuda()
exp_net.load_state_dict(torch.load('Chair_CLS_Expectation.pkl'))
exp_net.eval()

hidden_size = 100
rinf_output_size = int(cpv_size ** 2 / 2 + cpv_size / 2)
rinf_net = Model_Cov(2, 100, rinf_output_size).cuda()
rinf_net.train()
print("Number of parameters is:", sum(param.numel() for param in rinf_net.parameters()))

if torch.cuda.is_available():
    print("Cuda is ACTIVATED")
else:
    print("Cuda is DEACTIVATED")

number_of_epochs = 20
batch_size = 100
learning_rate = 0.001
crit = Cov_Loss.apply
opt = torch.optim.Adam(rinf_net.parameters(), lr=learning_rate)

# Network training
for epoch in range(number_of_epochs):

    permutation = torch.randperm(len(X))

    for i in range(0, len(X), batch_size):
        indices = permutation[i:i + batch_size]
        rand_tensors = x_np[indices]
        x_var = to_var(torch.tensor(rand_tensors).float(), requires_grad=True)
        gamma_var = to_var(torch.from_numpy(gamma_pre[indices]).float(), requires_grad=True)

        opt.zero_grad()

        out_exp = exp_net(x_var)
        out_r = rinf_net(x_var)

        loss = crit(gamma_var, out_exp, out_r)
        print("Loss at batch: ", i + batch_size, " is ", loss.item())

        loss.backward()
        opt.step()

    print("Loss at epoch ", epoch, " is: ", loss.item(), "\n\n")

    example = to_var(torch.Tensor([0.5, 0.5]))
    r_matrix_out = rinf_net(example)
    cov_matrix = torch.inverse( torch.matmul( torch.transpose(r_matrix_out, 0, 1), r_matrix_out ) )
    print(cov_matrix)

torch.save(rinf_net.state_dict(), 'Chair_CLS_Root_Inf_Matrix.pkl')