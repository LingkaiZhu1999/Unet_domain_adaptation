# File: loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
import numpy as np


class MultiClassDiceCELoss(nn.Module):
    def __init__(self, num_classes, weight=None):
        super(MultiClassDiceCELoss, self).__init__()
        # MONAI's DiceCELoss is perfect for this. It combines Dice and Cross-Entropy.
        # It needs logits as input.
        self.loss = DiceCELoss(
            to_onehot_y=True,  
            softmax=True,      # Apply softmax to model outputs (logits)
            # include_background=False, # Crucial: Don't calculate Dice for background
            batch=True,        # Calculate loss over the whole batch
            weight=weight      # Apply class weights
        )

    def forward(self, inputs, targets):
        # inputs shape: (B, C, H, W) - logits
        # targets shape: (B, C, H, W) - one-hot labels
        return self.loss(inputs, targets)


class DiceFocalLoss(nn.Module):
    """
    A robust combination of Dice and Focal Loss.
    Focal loss forces the model to focus on hard-to-classify examples.
    """
    def __init__(self, weight=None, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.dice = DiceLoss(to_onehot_y=False, softmax=True, include_background=False, batch=True, weight=weight)
        self.focal = FocalLoss(to_onehot_y=False, weight=weight, gamma=gamma) # `to_onehot_y=False` as our target is already one-hot

    def forward(self, inputs, targets):
        # inputs are logits [B, C, H, W], targets are one-hot [B, C, H, W]
        dice_loss = self.dice(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        
        # You can weigh the contribution of each loss
        total_loss = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return total_loss
    
# class MultiClassDiceCELoss(nn.Module):
#     """
#     Combination of Dice Loss and Cross Entropy Loss for multi-class segmentation.
#     """
#     def __init__(self, num_classes, weight_ce=1.0, weight_dice=1.0):
#         super(MultiClassDiceCELoss, self).__init__()
#         self.weight_ce = weight_ce
#         self.weight_dice = weight_dice
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.dice_loss = DiceLoss(softmax=True, batch=True)
#         self.num_classes = num_classes

#     def forward(self, inputs, targets):
#         # inputs shape: (B, C, H, W), targets shape: (B, H, W)
#         ce_val = self.ce_loss(inputs, targets)  # targets should be of shape (B, H, W) for CrossEntropyLoss
        
#         # The DiceLoss implementation handles this internally with softmax=True and to_onehot_y=True
#         dice_val = self.dice_loss(inputs, targets)
        
#         return self.weight_ce * ce_val + self.weight_dice * dice_val
    
class NTXentLoss(torch.nn.Module):

    def __init__(self, device, temperature, use_cosine_similarity, con_type='CL'):
        super(NTXentLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.con_type = con_type
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum") # sum all 2N terms of loss instead of getting mean val

    def _get_similarity_function(self, use_cosine_similarity):
        ''' Cosine similarity or dot similarity for computing loss '''
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)  # Use a small epsilon to avoid division by zero
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, batch_size):
        """
        An efficient pure PyTorch implementation to create the negative pair mask.
        """
        dim = 2 * batch_size
        
        # 1. Start with a mask where everything is a negative pair (all True)
        mask = torch.zeros((dim, dim), device=self.device, dtype=torch.bool)
        
        # 2. Set the main diagonal to False (to exclude self-pairs like (i, i))
        mask = mask.fill_diagonal_(True)
        
        mask = ~mask  
        # 3. Set the positive pair diagonals to False
        # Excludes pairs like (i, i+N) and (i+N, i)
            
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2) # extend the dimensions before calculating similarity 
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C), N input samples
        # y shape: (1, 2N, C), 2N output representations
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0)) # extend the dimensions before calculating similarity 
        return v

    def forward(self, zis, zjs):
        batch_size = zis.shape[0] # zis and zjs are both of shape [N, C]

        mask = self._get_correlated_mask(batch_size)
        representations = torch.cat([zjs, zis], dim=0) # [N, C] => [2N, C]

        similarity_matrix = self.similarity_function(representations, representations) # [2N, 2N]

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size) # upper diagonal, N x [left, right] positive sample pairs
        r_pos = torch.diag(similarity_matrix, -batch_size) # lower diagonal, N x [right, left] positive sample pairs
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1) # similarity of positive pairs, [2N, 1]

        negatives = similarity_matrix[mask].view(2 * batch_size, -1) # [2N, 2N]

        
        logits = torch.cat((positives, negatives), dim=1) # [2N, 2N+1], the 2N+1 elements of one column are used for one loss term
        logits /= self.temperature

        # labels are all 0, meaning the first value of each vector is the nominator term of CELoss
        # each denominator contains 2N+1-2 = 2N-1 terms, corresponding to all similarities between the sample and other samples.

        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)
    
    
# test nx loss
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = NTXentLoss(device=device, temperature=0.5, use_cosine_similarity=True)
    
    zis = torch.randn(4, 128).to(device)  # 4 samples, 128 features
    zjs = torch.randn(4, 128).to(device)  # 4 samples, 128 features
    
    loss = loss_fn(zis, zjs)
    print(f"NT-Xent Loss: {loss.item()}")