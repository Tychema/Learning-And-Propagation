import torch

from torch import tensor
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap


epoches=10000
L_num=200
torch.cuda.set_device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float32).to(device).view(1,1,3,3)
actions = torch.tensor([0, 1],dtype=torch.float32).to(device)
L = np.full((L_num, L_num), 0)
value_matrix = torch.tensor(L, dtype=torch.float32).to(device)
zeros_tensor = torch.zeros((1, 1, L_num, L_num),dtype=torch.float32).to(torch.float32)

class SPGG_Fermi(nn.Module):
    def __init__(self,epoches,L_num,device,r,K=0.1,count=0,cal_transfer=False):
        super(SPGG_Fermi, self).__init__()
        self.epoches=epoches
        self.L_num=L_num
        self.device=device
        self.r=r
        self.neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float32).to(device).view(1,1,3,3)
        self.K=K
        self.cal_transfer=cal_transfer
        self.count=count


    def profit_Matrix_to_Four_Matrix(self,profit_matrix,K):
        #calculate  W of the fermi update rule
        W_left=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,1))/K))
        W_right=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,-1,1))/K))
        W_up=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,0))/K))
        W_down=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,-1,0))/K))
        return W_left,W_right,W_up,W_down

    def fermiUpdate(self,type_t_matrix,profit_matrix,K):
        #W for four directions
        W_left,W_right,W_up,W_down=self.profit_Matrix_to_Four_Matrix(profit_matrix,0.1)
        #learning_direction
        learning_direction=torch.randint(0,4,(L_num,L_num)).to(device)
        # Whether learning_probabilities
        learning_probabilities=torch.rand(L_num,L_num).to(device)
        #fermiUpdate
        type_t1_matrix=(learning_direction==0)*((learning_probabilities<=W_left)*torch.roll(type_t_matrix,1,1)+(learning_probabilities>W_left)*type_t_matrix) +\
                          (learning_direction==1)*((learning_probabilities<=W_right)*torch.roll(type_t_matrix,-1,1)+(learning_probabilities>W_right)*type_t_matrix) +\
                            (learning_direction==2)*((learning_probabilities<=W_up)*torch.roll(type_t_matrix,1,0)+(learning_probabilities>W_up)*type_t_matrix) +\
                                (learning_direction==3)*((learning_probabilities<=W_down)*torch.roll(type_t_matrix,-1,0)+(learning_probabilities>W_down)*type_t_matrix)
        return type_t1_matrix.view(L_num,L_num)

    def pad_matrix(self,type_t_matrix):
        #Von Neumann neighborhoods and periodic boundaries
        tensor_matrix = torch.cat((type_t_matrix[-1:], type_t_matrix), dim=0)
        tensor_matrix = torch.cat((tensor_matrix[:, [-1]], tensor_matrix), dim=1)
        tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[1:2]), dim=0)
        tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[:, 1:2]), dim=1)
        return tensor_matrix

    def calculation_value(self,r,type_t_matrix):
        with torch.no_grad():
            #calculate the profit matrix

            pad_tensor = self.pad_matrix(type_t_matrix)
            d_matrix,c_matrix=self.type_matrix_to_three_matrix(pad_tensor)
            coorperation_matrix = c_matrix .view(1, 1, L_num+2, L_num+2).to(torch.float32)
            coorperation_num = torch.nn.functional.conv2d(coorperation_matrix, neibor_kernel,
                                                          bias=None, stride=1, padding=0).view(L_num,L_num).to(device)
            c_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r - 1)

            d_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r)
            c_5_profit_matrix = torch.nn.functional.conv2d(c_profit_matrix.view(1, 1, L_num+2, L_num+2), neibor_kernel,
                                                           bias=None, stride=1, padding=0).to(torch.float32).to(device)
            d_5_profit_matrix = torch.nn.functional.conv2d(d_profit_matrix.view(1, 1, L_num+2, L_num+2), neibor_kernel,
                                                           bias=None, stride=1, padding=0).to(device)
            d_matrix,c_matrix=self.type_matrix_to_three_matrix(type_t_matrix)
            profit_matrix = c_5_profit_matrix * c_matrix + d_5_profit_matrix * d_matrix
            return profit_matrix.view(L_num, L_num)



