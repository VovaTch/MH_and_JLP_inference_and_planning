import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from PIL import Image

def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
        print('Cuda is ON')
    return Variable(x, requires_grad=requires_grad)


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
        self.Softmax = nn.Softmax()

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

class ModelR(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, cpv_size=10):
        super(ModelR, self).__init__()
        self.Layer1 = nn.Linear(input_size, hidden_size)
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
                r_matrix[idx_x][self.cpv_size - idx_y - 1] = input[input_index]
                if idx_x is self.cpv_size - idx_y - 1 and input[input_index] <= 0:
                    r_matrix[idx_x][self.cpv_size - idx_y - 1] = - r_matrix[idx_x][self.cpv_size - idx_y - 1]
                input_index = input_index + 1

        return r_matrix

class ClsModel:

    def __init__(self):
        self.model = Model(2, 50, 10).cuda()
        self.model.load_state_dict(torch.load('Chair_CLS_Expectation.pkl'))
        self.model.cuda()
        self.check = 1

    def net_output(self, input_v):
        if len(input_v) is 6:
            input_v_small = input_v[3:5]
        else:
            input_v_small = input_v[0:2]
        torch.cuda.empty_cache()
        input_vector = torch.Tensor(input_v_small).cuda()
        #print("converting to torch")
        #input_variable = to_var(input_vector)
        output_variable = self.model(input_vector)
        return output_variable.tolist()

    def print_check(self):
        print('Damn SIGSEG...')

    def net_output_test(self, input_v):
        if len(input_v) is 6:
            input_v_small = input_v[3:5]
        else:
            input_v_small = input_v[0:2]
        output_variable = list()
        output_variable.append(0.5 * np.sin(input_v_small[0]) + 0.5)
        output_variable.append(0.5 - 0.5 * np.sin(input_v_small[0]))
        return output_variable

class ClsModelR:

    def __init__(self):
        self.model = ModelR(2, 100, 55).cuda()
        self.model.load_state_dict(torch.load('Chair_CLS_Root_Inf_Matrix.pkl'))
        self.model.cuda()
        self.check = 1

    def net_output(self, input_v):
        print('Net output call')
        if len(input_v) is 6:
            input_v_small = input_v[3:5]
        else:
            input_v_small = input_v[0:2] #MODIFY WHEN NETWORK IS READY
        torch.cuda.empty_cache()
        input_vector = torch.Tensor(input_v_small).cuda()
        #input_variable = to_var(input_vector)
        output_variable = self.model(input_vector)
        #print(output_variable)
        return output_variable.tolist()

    def net_output_test(self, input_v):
        if len(input_v) is 6:
            input_v_small = input_v[3:5]
        else:
            input_v_small = input_v[0:2]
        R_matrix = [[0.2 ,0.2],[0 ,0.2]]
        return R_matrix
