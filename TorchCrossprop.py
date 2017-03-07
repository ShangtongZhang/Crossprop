#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch
import torch.autograd.Variable
import torch.nn as nn
import torch.nn.functional as F

class TorchCrossprop(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(TorchCrossprop, self).__init__()
        self.fully_connected_1 = nn.Linear(dim_in, dim_hidden)
        self.fully_connected_2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x, target):
        net = self.fully_connected_1(x)
        phi = F.relu(net)
        y = self.fully_connected_2(phi)
        loss = nn.CrossEntropyLoss(y, target)


