import torch.nn as nn
from monai.losses import DiceLoss

import torch.nn as nn
from monai.losses import DiceLoss
import torch.nn.functional as F
import torch

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)
        self.ce = nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and the BCELoss in one single class. 

    def _loss(self, p, y):
        return self.dice(p, y) + self.ce(p, y.float())

    def forward(self, p, y):
        y_wt, y_tc, y_et = y[:, 0, :, :].unsqueeze(1), y[:, 1, :, :].unsqueeze(1), y[:, 2, :, :].unsqueeze(1) # labels for whole tumor, tumor core, and enhancing tumor
        p_wt, p_tc, p_et = p[:, 0, :, :].unsqueeze(1), p[:, 1, :, :].unsqueeze(1), p[:, 2, :, :].unsqueeze(1) # predictions
        l_wt, l_tc, l_et = self._loss(p_wt, y_wt), self._loss(p_tc, y_tc), self._loss(p_et, y_et) # losses 
        return l_wt + l_tc + l_et # sum over all losses 


# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super(BCEDiceLoss, self).__init__()

#     def forward(self, predict, target):
#         # bce = F.binary_cross_entropy_with_logits(predict, target)
#         smooth = 1e-5
#         predict = torch.sigmoid(predict)
#         num = target.size(0)
#         predict = predict.view(num, -1)
#         target = target.view(num, -1)
#         intersection = (predict * target)

#         dice = (2. * intersection.sum(1) + smooth) / (predict.sum(1) + target.sum(1) + smooth)
#         dice = 1 - dice.sum() / num
#         return dice
























# class Loss(nn.Module):
#     def __init__(self):
#         super(Loss, self).__init__()
#         self.dice = DiceLoss()
#         self.ce = nn.BCEWithLogitsLoss()

#     def _loss(self, p, y):
#         return self.dice(p, y) + self.ce(p, y.float())

#     def forward(self, p, y):
#         y_wt, y_tc, y_et = y > 0, ((y == 1) + (y == 3)) > 0, y == 3 # labels for whole tumor, tumor core, and enhancing tumor
#         p_wt, p_tc, p_et = p[:, 0].unsqueeze(1), p[:, 1].unsqueeze(1), p[:, 2].unsqueeze(1) # predictions
#         l_wt, l_tc, l_et = self._loss(p_wt, y_wt), self._loss(p_tc, y_tc), self._loss(p_et, y_et) # losses 
#         return l_wt + l_tc + l_et # sum over all losses 