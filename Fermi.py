
################################
##This package is a python source code of SPGG.

##Please see the following paper:

##Shen, Y.; Ma, Y.; Kang, H.; Sun, X.; Chen, Q.

##Propagation and Learning: Updating Strategies in Spatial Public Goods Games through Combined Fermi Update and Q-Learning. Chaos, Solitons & Fractals. 2024, 187, 115377.
################################



import torch

from torch import tensor
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap


colors = [(48, 83, 133), (218, 160, 90), (253, 243, 197)]  # R -> G -> B
colors = [(color[0] / 255, color[1] / 255, color[2] / 255) for color in colors]
cmap_mma = LinearSegmentedColormap.from_list("mma", colors, N=256)


colors = ["#eeeeee", "#111111", "#787ac0", ]
cmap = mpl.colors.ListedColormap(colors, N=3)

epoches=10000
L_num=200
torch.cuda.set_device("cuda:4" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

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


    def generated_default_type_matrix(self,L_num):
        probabilities = torch.tensor([1 / 2, 1 / 2])

        # 生成一个随机张量，其数值分别为0，1，2，根据指定的概率分布
        result_tensor = torch.multinomial(probabilities, L_num * L_num, replacement=True)
        result_tensor = result_tensor.view(L_num, L_num)
        return result_tensor.to(torch.float32).to(device)

    def generated_default_type_matrix2(self,L_num):
        tensor = torch.zeros(L_num, L_num)
        # 计算上半部分和下半部分的分界线（中间行）
        mid_row = L_num // 2
        # 将下半部分的元素设置为1
        tensor[mid_row:, :] = 1
        return tensor


    def profit_Matrix_to_Four_Matrix(self,profit_matrix,K):
        W_left=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,1))/K))
        W_right=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,-1,1))/K))
        W_up=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,1,0))/K))
        W_down=1/(1+torch.exp((profit_matrix-torch.roll(profit_matrix,-1,0))/K))
        return W_left,W_right,W_up,W_down

    def fermiUpdate(self,type_t_matrix,profit_matrix,K):

        W_left,W_right,W_up,W_down=self.profit_Matrix_to_Four_Matrix(profit_matrix,0.1)

        learning_direction=torch.randint(0,4,(L_num,L_num)).to(device)

        learning_probabilities=torch.rand(L_num,L_num).to(device)

        type_t1_matrix=(learning_direction==0)*((learning_probabilities<=W_left)*torch.roll(type_t_matrix,1,1)+(learning_probabilities>W_left)*type_t_matrix) +\
                          (learning_direction==1)*((learning_probabilities<=W_right)*torch.roll(type_t_matrix,-1,1)+(learning_probabilities>W_right)*type_t_matrix) +\
                            (learning_direction==2)*((learning_probabilities<=W_up)*torch.roll(type_t_matrix,1,0)+(learning_probabilities>W_up)*type_t_matrix) +\
                                (learning_direction==3)*((learning_probabilities<=W_down)*torch.roll(type_t_matrix,-1,0)+(learning_probabilities>W_down)*type_t_matrix)
        return type_t1_matrix.view(L_num,L_num)

    def pad_matrix(self,type_t_matrix):

        tensor_matrix = torch.cat((type_t_matrix[-1:], type_t_matrix), dim=0)

        tensor_matrix = torch.cat((tensor_matrix[:, [-1]], tensor_matrix), dim=1)

        tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[1:2]), dim=0)

        tensor_matrix = torch.cat((tensor_matrix, tensor_matrix[:, 1:2]), dim=1)
        return tensor_matrix

    def calculation_value(self,r,type_t_matrix):
        with torch.no_grad():

            pad_tensor = self.pad_matrix(type_t_matrix)
            d_matrix,c_matrix=self.type_matrix_to_three_matrix(pad_tensor)
            coorperation_matrix = c_matrix .view(1, 1, L_num+2, L_num+2).to(torch.float32)
            # 下面这个卷积占了一轮的大部分时间约1秒钟，但是其他卷积都是一瞬间完成的，不知道为什么
            coorperation_num = torch.nn.functional.conv2d(coorperation_matrix, neibor_kernel,
                                                          bias=None, stride=1, padding=0).view(L_num,L_num).to(device)
            # c和r最后的-1是最开始要贡献到池里面的1
            c_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r - 1)

            d_profit_matrix = self.pad_matrix((coorperation_num) / 5 * r)
            c_5_profit_matrix = torch.nn.functional.conv2d(c_profit_matrix.view(1, 1, L_num+2, L_num+2), neibor_kernel,
                                                           bias=None, stride=1, padding=0).to(torch.float32).to(device)
            d_5_profit_matrix = torch.nn.functional.conv2d(d_profit_matrix.view(1, 1, L_num+2, L_num+2), neibor_kernel,
                                                           bias=None, stride=1, padding=0).to(device)
            d_matrix,c_matrix=self.type_matrix_to_three_matrix(type_t_matrix)
            profit_matrix = c_5_profit_matrix * c_matrix + d_5_profit_matrix * d_matrix
            return profit_matrix.view(L_num, L_num)


    def save_shot_data(self,type_t_matrix: tensor,i,r,profit_matrix,generated):

        self.mkdir('data/Fermi/shot_pic/r={}/two_type/{}/type_t_matrix'.format(r,str(generated)))
        np.savetxt('data/OFermi/shot_pic/r={}/two_type/{}/type_t_matrix/{}_r={}_epoches={}_L={}_T={}_第{}次实验数据.txt'.format(str(r),str(generated),"type_t_matrix", str(r), str(self.epoches), str(self.L_num), str(i), str(self.count)),type_t_matrix.cpu().numpy())
        self.mkdir('data/Fermi/shot_pic/r={}/two_type/{}/profit_matrix'.format(r,str(generated)))
        np.savetxt('data/Fermi/shot_pic/r={}/two_type/{}/profit_matrix/{}_r={}_epoches={}_L={}_T={}_第{}次实验数据.txt'.format(str(r),str(generated),"profit_matrix", str(r), str(self.epoches), str(self.L_num), str(i), str(self.count)),profit_matrix.cpu().numpy())


    def cal_fra_and_value(self, D_Y, C_Y, D_Value, C_Value,all_value, type_t_matrix, d_matrix, c_matrix, profit_matrix,i):
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
        all_value = np.append(all_value,profit_matrix.sum().item())
        return  D_Y, C_Y, D_Value, C_Value, count_0, count_1, all_value

    def mkdir(self,path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)

    def save_data(self,type,name,r,count,data):
        self.mkdir('data/Fermi/generated2/'+str(type))
        try:
            np.savetxt('data/Fermi/generated2/{}/{}_r={}_epoches={}_L={}_{}times.txt'.format(str(type), name, str(r), str(self.epoches), str(self.L_num),str(count)), data)
        except:
            print("Save failed")

    def run_line2_pic(self, loop_num1=51, loop_num2=10):
        r = 0
        for j in range(loop_num1):
            for i in range(loop_num2):
                r1=r/10
                print("loop_num1: " + str(j) + " loop_num2: " + str(i) + "r=" + str(r1))
                D_Y, C_Y, D_Value, C_Value, type_t_matrix, count_0, count_1, all_value= self.run(r1, generated="generated1")
                self.save_data('C_fra', 'C_fra', r1, i, C_Y)
                self.save_data('D_fra', 'D_fra', r1, i, D_Y)
                self.save_data('C_value', 'C_value', r1, i, C_Value)
                self.save_data('D_value', 'D_value', r1, i, D_Value)
                self.save_data('all_value', 'all_value', r1, i, all_value)
            r = r + 1

    def line1_pic(self, r,generated="generated1"):
        D_Y_ave, C_Y_ave, D_Value_ave, C_Value_ave,all_value_ave, type_t_matrix, count_0_ave, count_1_ave = np.zeros(epoches + 1), np.zeros(epoches + 1), np.zeros(epoches),np.zeros(epoches), np.zeros(epoches), np.zeros(epoches), 0, 0
        loop_num = 1
        for i in range(loop_num):
            D_Y, C_Y, D_Value, C_Value, type_t_matrix, count_0, count_1, all_value = self.run(r, generated=generated)








