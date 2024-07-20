


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
epsilon=-1
neibor_kernel=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=torch.float64).to(device).view(1,1,3,3)
actions = torch.tensor([0, 1],dtype=torch.float64).to(device)

zeros_tensor = torch.zeros((1, 1, L_num, L_num),dtype=torch.float64).to(torch.float64)

class SPGG_Qlearning(nn.Module):
    def __init__(self,L_num,device,alpha,gamma,epsilon,r,epoches,lr=0.2,eta=0.8,count=0,cal_transfer=False):
        super(SPGG_Qlearning, self).__init__()
        self.epoches=epoches
        self.L_num=L_num
        self.device=device
        self.alpha=alpha
        self.r=r
        self.gamma=gamma
        self.epsilon=epsilon
        self.cal_transfer=cal_transfer
        self.lr=lr
        self.eta=eta
        self.count=count


    def indices_Matrix_to_Four_Matrix(self,indices):
        indices_left=torch.roll(indices,1,1)
        indices_right=torch.roll(indices,-1,1)
        indices_up=torch.roll(indices,1,0)
        indices_down=torch.roll(indices,-1,0)
        return indices_left,indices_right,indices_up,indices_down

    #update Qtable
    def updateQMatrix(self,alpha,gamma,type_t_matrix: tensor, type_t1_matrix: tensor, Q_tensor: tensor, profit_matrix: tensor):
        C_indices = torch.arange(type_t_matrix.numel()).to(device)

        A_indices = type_t_matrix.view(-1).long()

        B_indices = type_t1_matrix.view(-1).long()

        max_values, _ = torch.max(Q_tensor[C_indices, B_indices], dim=1)

        update_values = (1 - self.eta) * Q_tensor[C_indices, A_indices, B_indices] + self.eta * (profit_matrix.view(-1) + gamma * max_values)

        Q_tensor[C_indices, A_indices, B_indices] = update_values
        return Q_tensor

    def profit_Matrix_to_Four_Matrix(self,profit_matrix,K):
        W_left=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,1))/K))
        W_right=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,-1,1))/K))
        W_up=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,0))/K))
        W_down=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,-1,0))/K))
        return W_left,W_right,W_up,W_down


    def pad_matrix(self,type_t_matrix):

        tensor_matrix = torch.cat((type_t_matrix[-1:], type_t_matrix), dim=0)

        tensor_matrix = torch.cat((tensor_matrix[:, [-1]], tensor_matrix), dim=1)

        tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[1:2]), dim=0)

        tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[:, 1:2]), dim=1)
        return tensor_matrix


    #calculate the profit matrix
    def calculation_value(self,r,type_t_matrix):
        with torch.no_grad():


            pad_tensor = self.pad_matrix(type_t_matrix)
            d_matrix,c_matrix=self.type_matrix_to_three_matrix(pad_tensor)
            coorperation_matrix = c_matrix .view(1, 1, L_num+2, L_num+2).to(torch.float64)

            coorperation_num = torch.nn.functional.conv2d(coorperation_matrix, neibor_kernel,
                                                          bias=None, stride=1, padding=0).view(L_num,L_num).to(device)

            c_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r - 1)

            d_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r)
            c_5_profit_matrix = torch.nn.functional.conv2d(c_profit_matrix.view(1, 1, L_num+2, L_num+2), neibor_kernel,
                                                           bias=None, stride=1, padding=0).to(torch.float64).to(device)
            d_5_profit_matrix = torch.nn.functional.conv2d(d_profit_matrix.view(1, 1, L_num+2, L_num+2), neibor_kernel,
                                                           bias=None, stride=1, padding=0).to(device)
            d_matrix,c_matrix=self.type_matrix_to_three_matrix(type_t_matrix)

            profit_matrix = c_5_profit_matrix * c_matrix + d_5_profit_matrix * d_matrix
            return profit_matrix.view(L_num, L_num).to(torch.float64)


    def type_matrix_to_three_matrix(self,type_matrix: tensor):

        d_matrix = torch.where(type_matrix == 0, torch.tensor(1), torch.tensor(0)).to(device)
        c_matrix = torch.where(type_matrix == 1, torch.tensor(1), torch.tensor(0)).to(device)
        return d_matrix, c_matrix


    def generated_default_type_matrix(self):
        probabilities = torch.tensor([1 /2, 1 / 2])


        result_tensor = torch.multinomial(probabilities, L_num * L_num, replacement=True)
        result_tensor = result_tensor.view(L_num, L_num)
        return result_tensor.to(torch.float64).to("cpu")

    def generated_default_type_matrix2(self):
        tensor = torch.zeros(L_num, L_num)

        mid_row = L_num // 2

        tensor[mid_row:, :] = 1
        return tensor

    def generated_default_type_matrix3(self):
        tensor = torch.zeros(L_num, L_num)

        return tensor


    def c_mean_v(self,value_tensor):
        positive_values = value_tensor[value_tensor > 0.0]

        mean_of_positive = torch.mean(positive_values)
        return mean_of_positive.item() + 1


    def c_mean_v2(self,value_tensor):

        positive_num = (value_tensor > 0).to(device)
        negetive_num = (value_tensor < 0).to(device)

        mean_of_positive_elements = (value_tensor.to(torch.float64).sum()) / ((positive_num + negetive_num).sum())
        return mean_of_positive_elements.to("cpu")


    def shot_pic(self,type_t_matrix: tensor,i,r,profit_matrix):
        plt.clf()
        plt.close("all")

        fig = plt.figure(figsize=(40,40))
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 2)

        color_map = {

            0: (0, 0, 0),
            1: (255, 255, 255),
        }
        image = np.zeros((L_num, L_num, 3), dtype=np.uint8)
        for label, color in color_map.items():
            image[type_t_matrix.cpu() == label] = color

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        plt.imshow(image,interpolation='None')
        plt.imshow(image,interpolation='None')
        plt.clf()
        plt.close("all")


    def shot_pic2(self,type_t_matrix: tensor,i,r):
        plt.clf()
        plt.close("all")

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 4)
        color_map = {

            0:(128, 128, 128),

            1:(255, 255, 255),

            2:(0, 0, 0),

            3:(31,119,180)
        }
        image = np.zeros((L_num, L_num, 3), dtype=np.uint8)
        for label, color in color_map.items():
            image[type_t_matrix.cpu() == label] = color
        plt.title('Qlearning: '+f"T:{i}")
        plt.imshow(image,interpolation='None')

        plt.clf()
        plt.close("all")

    def shot_save_data(self,type_t_minus_matrix: tensor,type_t_matrix: tensor,type_t1_matrix: tensor,i,r,profit_matrix,Q_matrix):

        C_indices = torch.arange(type_t_matrix.numel()).to(device)

        A_indices = type_t_minus_matrix.view(-1).long()

        B_indices = type_t_matrix.view(-1).long()
        Q_sa_matrix = Q_matrix[C_indices, A_indices, B_indices].view(L_num, L_num)


    def cal_fra_and_value(self, D_Y, C_Y, D_Value, C_Value,all_value, type_t_minus_matrix,type_t_matrix, d_matrix, c_matrix, profit_matrix,i):


        d_value = d_matrix * profit_matrix
        c_value = c_matrix * profit_matrix
        dmean_of_positive = self.c_mean_v2(d_value)
        cmean_of_positive = self.c_mean_v2(c_value)
        count_0 = torch.sum(type_t_matrix == 0).item()
        count_1 = torch.sum(type_t_matrix == 1).item()
        D_Y = np.append(D_Y, count_0 / (L_num * L_num))
        C_Y = np.append(C_Y, count_1 / (L_num * L_num))
        D_Value = np.append(D_Value, dmean_of_positive)
        C_Value = np.append(C_Value, cmean_of_positive)
        all_value = np.append(all_value, profit_matrix.sum().item())
        CC, DD, CD, DC = self.cal_transfer_num(type_t_minus_matrix,type_t_matrix)
        return  D_Y, C_Y, D_Value, C_Value,all_value, count_0, count_1, CC, DD, CD, DC


    def cal_transfer_num(self,type_t_matrix,type_t1_matrix):
        CC=(torch.where((type_t_matrix==1)&(type_t1_matrix==1),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        DD=(torch.where((type_t_matrix==0)&(type_t1_matrix==0),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        CD=(torch.where((type_t_matrix==1)&(type_t1_matrix==0),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        DC=(torch.where((type_t_matrix==0)&(type_t1_matrix==1),torch.tensor(1),torch.tensor(0)).sum().item())/ (L_num * L_num)
        return CC,DD,CD,DC


    def extract_Qtable(self,Q_tensor, type_t_matrix):
        C_indices = torch.where(type_t_matrix.squeeze() == 1)[0]
        D_indices = torch.where(type_t_matrix.squeeze() == 0)[0]
        C_Q_table = Q_tensor[C_indices]
        D_indices = Q_tensor[D_indices]
        C_q_mean_matrix = torch.mean(C_Q_table, dim=0)
        D_q_mean_matrix = torch.mean(D_indices, dim=0)
        return D_q_mean_matrix.cpu().numpy(), C_q_mean_matrix.cpu().numpy()

    def split_four_policy_type(self,Q_matrix):
        CC = torch.where((Q_matrix[:, 1, 1] > Q_matrix[:, 1, 0]) & (
                Q_matrix[:, 0, 0] <= Q_matrix[:, 0, 1]), torch.tensor(1), torch.tensor(0))
        DD = torch.where((Q_matrix[:, 0, 0] > Q_matrix[:, 0, 1]) & (
                    Q_matrix[:, 1, 1] <= Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0))
        CDC = torch.where((Q_matrix[:, 0, 0] < Q_matrix[:, 0, 1]) & (Q_matrix[:, 1, 1] < Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0))
        StickStrategy=torch.where((Q_matrix[:,0,0]>Q_matrix[:,0,1])&(Q_matrix[:,1,1]>Q_matrix[:,1,0]),torch.tensor(1),torch.tensor(0))
        return DD.view((L_num,L_num)),CC.view((L_num,L_num)), CDC.view((L_num,L_num)), StickStrategy.view((L_num,L_num))

    def split_five_policy_type(self,Q_matrix,type_t_matrix):
        CC = torch.where((Q_matrix[:, 1, 1] > Q_matrix[:, 1, 0]) & (
                Q_matrix[:, 0, 0] <= Q_matrix[:, 0, 1]), torch.tensor(1), torch.tensor(0)).view((L_num,L_num))
        DD = torch.where((Q_matrix[:, 0, 0] > Q_matrix[:, 0, 1]) & (
                    Q_matrix[:, 1, 1] <= Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0)).view((L_num,L_num))
        CDC = torch.where((Q_matrix[:, 0, 0] < Q_matrix[:, 0, 1]) & (Q_matrix[:, 1, 1] < Q_matrix[:, 1, 0]), torch.tensor(1), torch.tensor(0)).view((L_num,L_num))
        StickStrategy=torch.where((Q_matrix[:,0,0]>Q_matrix[:,0,1])&(Q_matrix[:,1,1]>Q_matrix[:,1,0]),torch.tensor(1),torch.tensor(0)).view((L_num,L_num))
        CDC_C=CDC*torch.where(type_t_matrix==1,torch.tensor(1),torch.tensor(0))
        CDC_D=CDC*torch.where(type_t_matrix==0,torch.tensor(1),torch.tensor(0))
        CDC_neibor_num=0
        other_neibor_num=0
        CDC_neibor_DD, CDC_neibor_CC=torch.zeros((L_num,L_num)).to(device),torch.zeros((L_num,L_num)).to(device)
        if CDC.sum().item()!=0:
            CDC_neibor_matrix=self.pad_matrix(CDC.to(torch.float64).to(device))
            CDC_neibor_conv2d = torch.nn.functional.conv2d(CDC_neibor_matrix.view(1,1,L_num+2,L_num+2), neibor_kernel,
                                                          bias=None, stride=1, padding=0).view(L_num,L_num).to(device)
            CDC_neibor_num=(CDC_neibor_conv2d*CDC).sum().item()/CDC.sum().item()
            other_neibor_num = (CDC_neibor_conv2d * (1-CDC)).sum().item() / (1-CDC).sum().item()
            CDC_neibor_DD=torch.where(CDC_neibor_conv2d*(1-CDC)>0,torch.tensor(1),torch.tensor(0))*DD
            CDC_neibor_CC=torch.where(CDC_neibor_conv2d*(1-CDC)>0,torch.tensor(1),torch.tensor(0))*CC
        return DD,CC, CDC, StickStrategy,CDC_D,CDC_C,CDC_neibor_num,other_neibor_num,CDC_neibor_DD,CDC_neibor_CC

    def cal_four_type_value(self,DD,CC,CDC,StickStrategy,profit_matrix):
        CC_value = profit_matrix * CC
        DD_value = profit_matrix * DD
        CDC_value = profit_matrix * CDC
        StickStrategy_value = profit_matrix * StickStrategy
        return  DD_value,CC_value, CDC_value, StickStrategy_value

    def cal_five_type_value(self,DD,CC,CDC,StickStrategy,CDC_D,CDC_C,CDC_neibor_DD,CDC_neibor_CC,profit_matrix):
        CC_value = profit_matrix * CC
        DD_value = profit_matrix * DD
        CDC_value = profit_matrix * CDC
        StickStrategy_value = profit_matrix * StickStrategy
        CDC_C_value = profit_matrix * CDC_C
        CDC_D_value = profit_matrix * CDC_D
        CDC_neibor_DD_value = profit_matrix * CDC_neibor_DD
        CDC_neibor_CC_value = profit_matrix * CDC_neibor_CC
        return  DD_value,CC_value, CDC_value, StickStrategy_value,CDC_D_value,CDC_C_value,CDC_neibor_DD_value,CDC_neibor_CC_value


    def mkdir(self,path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)

    def save_data(self,type,name,r,count,data):
        self.mkdir('data/'+str(type))
        np.savetxt('data/{}/{}_r={}_epoches={}_L={}_{}.txt'.format(str(type), name,str(r),str(self.epoches),str(self.L_num),str(count)), data)

    def run_line2_pic(self, loop_num1= 51, loop_num2 = 10):
        r=0
        for j in range(loop_num1):
            for i in range(loop_num2):
                r1=r/10
                print("loop_num1: "+str(j)+" loop_num2: "+str(i)+" r="+str(r1))
                self.count=i
                D_Y, C_Y, D_Value, C_Value, all_value, Q_matrix, type_t_matrix, count_0, count_1, \
                CC_data, DD_data, CD_data, DC_data, DD_Y, CC_Y, CDC_Y, StickStrategy_Y, DD_value_np, CC_value_np, CDC_value_np, StickStrategy_value_np = self.run(r1, self.alpha, self.gamma, self.epsilon, self.epoches, self.L_num, self.device, type="line1")
            r=r+1




    def extra_Q_table(self,loop_num):
        for i in range(loop_num):
            Q_matrix,type_t_matrix = self.run(self.r, self.alpha,self.gamma,self.epsilon,self.epoches, self.L_num,self.device,type="Qtable")
            D_q_mean_matrix, C_q_mean_matrix = self.extract_Qtable(Q_matrix, type_t_matrix)
            print(D_q_mean_matrix,C_q_mean_matrix)
            self.save_data('D_Qtable', 'D_Qtable',self.r, str(i), D_q_mean_matrix)
            self.save_data('C_Qtable', 'C_Qtable',self.r, str(i), C_q_mean_matrix)



    def line1_pic(self, r):
        loop_num = 1
        for i in range(loop_num):
            print("第i轮:", i)
            self.count = i
            D_Y, C_Y, D_Value, C_Value, all_value, Q_matrix, type_t_matrix, count_0, count_1,CC_data, DD_data, CD_data, DC_data, \
            Q_D_DD, Q_D_DC, Q_D_CD, Q_D_CC,Q_C_DD, Q_C_DC, Q_C_CD, Q_C_CC = self.run(self.r, self.alpha, self.gamma, self.epsilon, self.epoches, self.L_num, self.device, type="line1")







