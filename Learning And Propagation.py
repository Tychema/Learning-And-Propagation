


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




