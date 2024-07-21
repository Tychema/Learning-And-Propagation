
################################
##This package is a python source code of SPGG.

##Please see the following paper:

##Shen, Y.; Ma, Y.; Kang, H.; Sun, X.; Chen, Q.

##Propagation and Learning: Updating Strategies in Spatial Public Goods Games through Combined Fermi Update and Q-Learning
################################

import torch

from torch import tensor
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

L_num=200
torch.cuda.set_device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha=0.8
gamma=0.8
epsilon=0.02
neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float32).to(device).view(1,1,3,3)
actions = torch.tensor([0, 1],dtype=torch.float32).to(device)
L = np.full((L_num, L_num), 0)

class SPGG_Qlearning(nn.Module):
    def __init__(self,L_num,device,alpha,gamma,epsilon,r,epoches,count=0,cal_transfer=False):
        super(SPGG_Qlearning, self).__init__()
        self.epoches=epoches
        self.L_num=L_num
        self.device=device
        self.alpha=alpha
        self.r=r
        self.gamma=gamma
        self.epsilon=epsilon
        self.cal_transfer=cal_transfer
        self.count=count

    #Qtable update
    def updateQMatrix(self,alpha,gamma,type_t_matrix: tensor, type_t1_matrix: tensor, Q_tensor: tensor, profit_matrix: tensor):
        C_indices = torch.arange(type_t_matrix.numel()).to(device)
        A_indices = type_t_matrix.view(-1).long()
        B_indices = type_t1_matrix.view(-1).long()
        max_values, _ = torch.max(Q_tensor[C_indices, B_indices], dim=1)
        update_values = Q_tensor[C_indices, A_indices, B_indices] + alpha * (profit_matrix.view(-1) + gamma * max_values - Q_tensor[C_indices, A_indices, B_indices])
        Q_tensor[C_indices, A_indices, B_indices] = update_values
        return Q_tensor

    #Von Neumann neighborhoods and periodic boundaries
    def pad_matrix(self,type_t_matrix):

        tensor_matrix = torch.cat((type_t_matrix[-1:], type_t_matrix), dim=0)

        tensor_matrix = torch.cat((tensor_matrix[:, [-1]], tensor_matrix), dim=1)

        tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[1:2]), dim=0)

        tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[:, 1:2]), dim=1)
        return tensor_matrix

    #use Q-table update type_matrix
    def type_matrix_change(self,epsilon,type_matrix: tensor, Q_matrix: tensor):
        indices = type_matrix.long().flatten()
        Q_probabilities = Q_matrix[torch.arange(len(indices)), indices]

        max_values, _ = torch.max(Q_probabilities, dim=1)

        max_tensor = torch.where(Q_probabilities == max_values[:, None], torch.tensor(1.0, device=device),
                                 torch.tensor(0.0, device=device))

        rand_tensor = torch.rand(max_tensor.size()).to(device)

        masked_tensor = (max_tensor.float() - (1 - max_tensor.float()) * 1e9).to(device)

        sum_tensor = (masked_tensor + rand_tensor).to(device)

        indices = torch.argmax(sum_tensor, dim=1).to(device)

        random_type = torch.randint(0,2, (L_num, L_num)).to(device)

        mask = (torch.rand(L_num, L_num) >= epsilon).long().to(device)

        updated_values = mask.flatten().unsqueeze(1) * indices.unsqueeze(1) + (1 - mask.flatten().unsqueeze(1)) * random_type.flatten().float().unsqueeze(1)

        updated_tensor = updated_values.view(L_num, L_num).to(device)
        return updated_tensor

    # calculate the profit matrix
    def calculation_value(self,r,type_t_matrix):
        with torch.no_grad():


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

