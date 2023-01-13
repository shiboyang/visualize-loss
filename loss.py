# @Time    : 2023/1/10 上午10:57
# @Author  : Boyang
# @Site    : 
# @File    : loss.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, inputs, target):
        out = F.log_softmax(inputs)
        out = F.nll_loss(out, target)
        return out


class RingLoss(nn.Module):
    def __init__(self, scale, r):
        super(RingLoss, self).__init__()
        self.scale = torch.as_tensor(scale)
        self.r = nn.Parameter(torch.as_tensor(r))

    def forward(self, inputs, target):
        out = torch.linalg.norm(inputs, dim=1)
        out = self.scale / 2 * torch.mean((out - self.r) ** 2)

        return out
