from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import os
import numpy as np
# Dataset

class BratsTrainDataset(Dataset):
    def __init__(self, datapath='/mnt/asgard2/data/lingkai/braTS20/slice/', augmentation=None):
        self.augmentation = augmentation
        self.folderpaths = {
            'mask': os.path.join(datapath, 'label/'),
            'flair': os.path.join(datapath, 'flair/')
        }
        self.flair_path = []
        self.label_path = []
        for files in os.listdir(self.folderpaths['mask']):
            flair_path = os.path.join(self.folderpaths['flair'], files)
            label_path = os.path.join(self.folderpaths['mask'], files)
            self.flair_path.append(flair_path)
            self.label_path.append(label_path)

    
    def __getitem__(self, idx):
        images = {}
        
        images['flair'] = np.load(self.flair_path[idx])
        images['mask'] = np.load(self.label_path[idx])
        
        if self.augmentation:
            images = self.augmentaion(image=images['flair'], mask=images['labels'])

            images['flair'] = images['flair']

        for name in images:
            images[name] = torch.from_numpy(images[name])

        # stack modalities 
        # input = torch.stack([images['t1'], images['t1ce'], images['t2'], images['flair']], dim=0)
        input = images['flair']
        # map pixels with value of 4 to 3
        images['mask'][images['mask']==4] = 3
        
        # one-hot encode ground truth
        # images['mask'] = F.one_hot(images['mask'].long().unsqueeze(0), num_classes=4).permute(0, 3, 1, 2).contiguous().squeeze(0)

        return input.float().unsqueeze(0), images['mask'].long().unsqueeze(0)

    def __len__(self):
        return len(os.listdir(self.folderpaths['mask'])) - 1