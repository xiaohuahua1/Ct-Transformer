from torch import nn
import torch
import math
import torch.nn.functional as F
from args import *
from data_process import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuantileLoss(nn.Module):

    def __init__(self, quantiles):
        ##takes a list of quantiles
        super().__init__()
        self.quantiles = quantiles
        # self.loss_func = nn.SmoothL1Loss(reduction='none').to(device)

    def forward(self, preds, target):
        # assert not target.requires_grad
        # assert preds.size(0) == target.size(0)#检验程序使用的，如果不满足条件，程序会自动退出
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            # errors = torch.pow(errors,2)
            # errors = self.loss_func(preds[:,i],target)
            losses.append(torch.max((q-1) * errors,q * errors ).unsqueeze(1))

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        # print(loss.shape)
        # print(' ')
        return loss
    
class PretrainLoss(nn.Module):
    def __init__(self):
        ##takes a list of quantiles
        super().__init__()
    def forward(self, preds, target,mask):
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss