# File: metrics.py
import torch
from monai.metrics import DiceMetric

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val 
        self.count += 1
        self.avg = self.sum / self.count