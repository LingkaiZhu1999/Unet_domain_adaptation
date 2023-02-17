import torch.nn as nn
from monai.losses import DiceLoss

import torch.nn as nn
import torch.nn.functional as F
import torch

class LogDiceLoss(nn.Module):
    def __init__(self):
        super(LogDiceLoss, self).__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)

    def _loss(self, p, y):
        return torch.log(self.dice(p, y))

    def forward(self, p, y):
        y_wt, y_tc, y_et = y[:, 0, :, :].unsqueeze(1), y[:, 1, :, :].unsqueeze(1), y[:, 2, :, :].unsqueeze(1) # labels for whole tumor, tumor core, and enhancing tumor
        p_wt, p_tc, p_et = p[:, 0, :, :].unsqueeze(1), p[:, 1, :, :].unsqueeze(1), p[:, 2, :, :].unsqueeze(1) # predictions
        l_wt, l_tc, l_et = self._loss(p_wt, y_wt), self._loss(p_tc, y_tc), self._loss(p_et, y_et) # losses 
        return l_wt + l_tc + l_et # sum over all losses 