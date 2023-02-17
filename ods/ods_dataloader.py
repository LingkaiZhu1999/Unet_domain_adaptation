from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

from torch.utils import data
from torchvision import transforms 
import albumentations as A
import cv2

def augment_data(image, tissue_mask):
    """
    Data augmentation
    Flip image and mask. Rotate image and mask.
    """

    image = np.array(image) 
    tissue_mask = np.array(tissue_mask)
    
    if np.random.rand() > 0.5:
        ax = np.random.choice([0, 1])
        image = np.flip(image, axis=ax)
        tissue_mask = np.flip(tissue_mask, axis=ax)	
    
    if np.random.rand() > 0.5:
        rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
        image = np.rot90(image, k=rot, axes=[0, 1])
        tissue_mask = np.rot90(tissue_mask, k=rot, axes=[0, 1])
        
    return image, tissue_mask

class BaseDataset(data.Dataset):
    def __init__(self,
                mode,
                split_path,
                dataset=None,
                split=None,
                ):
        """

        Args:
            mode (string): Train/val/test.
            split_path (string): Path to train/val/test split.
            dataset (str, optional): Dataset name. Defaults to None.
            split (str, optional): Split. Defaults to None.
        """
        self.mode = mode
        self.dataset = dataset.lower() if dataset is not None else None
        self.split = split.lower() if split is not None else None
        
        split = pd.read_csv(split_path)[self.mode]
        split = split.dropna().reset_index(drop=True)
        self.split_data = split
        assert len(split) > 0, "Split should not be empty"

    def __len__(self):
        return None

    def __getitem__(self, idx):
        return None

class ImageDataset(BaseDataset):
    def __init__(self,
                dataset,
                image_path,
                mask_path,
                test=False,
                augmentation=None,
                pair_gen = False,
                val_source = False,
                **kwargs
        ):
        """

        Args:
            image_path (string): Path to the images.
            anno_path (string): Path to the annotations. 
            mask_path (string): Path to the tussue masks.
            augmentation (bool, optional): Augmentation. Defaults to False.
        """
        super(ImageDataset, self).__init__(**kwargs)
        self.image_path = image_path
        # self.anno_path = anno_path
        self.mask_path = mask_path
        self.test = test
        self.augmentation = augmentation
        self.pair_gen = pair_gen

        self.image_names = self.split_data
        self.label_names = self.split_data
        self.val_source = val_source
        self.dataset = dataset
        if dataset == 'refuge':
            self.mapping = {(0, 0, 0): 0, # 0 = background
                            (255, 255, 0): 1, # 1 = class 1, cup
                            (255, 0, 0): 2 # 2 = class2, disc
                            }
        elif dataset == 'idrid':
            self.mapping = {(255, 255, 0): 0, # 0 = background
                            (0, 0, 0): 1, # 1 = class 1, disc
                            }
        elif dataset == 'rimone':
            self.mapping = {(0, 0, 0): 0, # 0 = background
                            (255, 255, 255): 1, # 1 = class 1, disc
                            }
        else:
            raise ValueError('Dataset not specified by the dataloader file, please add the dataset.')
        print(f"Total number of {self.mode} images: {len(self.image_names)}")

    def __len__(self):
        return len(self.image_names)

    def mask_to_class_rgb(self, mask):
        mask = torch.from_numpy(mask)
        mask_out = torch.empty(mask.shape[0], mask.shape[1], dtype=torch.long)
        for k in self.mapping:
            idx = (mask == torch.tensor(k, dtype=torch.uint8))
            idx = (idx.sum(-1) == 3)
            mask_out[idx] = torch.tensor(self.mapping[k], dtype=torch.long)

        mask_out[mask_out == 2] = 1 
        return mask_out.unsqueeze(0)



    def __getitem__(self, idx):
        image_name = self.image_names[idx] 
        label_name = self.label_names[idx]

        image = np.array(Image.open(os.path.join(self.image_path, image_name)))
        # annotation = Image.open(os.path.join(self.anno_path, image_name))
        mask = np.array(Image.open(os.path.join(self.mask_path, label_name)).convert('RGB'))

        if self.pair_gen:
            if not self.val_source:
                transformed1 = self.augmentation(image=image, mask=mask)
                image1, mask1 = transformed1['image'], transformed1['mask']
                transformed2 = self.augmentation(image=image, mask=mask)
                image2, mask2 = transformed2['image'], transformed2['mask']
                # return transforms.ToTensor()(image1), transforms.ToTensor()(image2),  \
                # self.mask_to_class_rgb(np.array(mask1)), self.mask_to_class_rgb(mask2)
                return transforms.ToTensor()(image1), transforms.ToTensor()(image2),  \
            self.mask_to_class_rgb(mask1), self.mask_to_class_rgb(mask2)
            else:
                transformed1 = self.augmentation(image=image, mask=mask)
                image1, mask1 = transformed1['image'], transformed1['mask']
                transformed2 = self.augmentation(image=image, mask=mask)
                image2, mask2 = transformed2['image'], transformed2['mask']
                # return transforms.ToTensor()(image1), transforms.ToTensor()(image2),  \
                # self.mask_to_class_rgb(np.array(mask1)), self.mask_to_class_rgb(mask2)
                return transforms.ToTensor()(image), self.mask_to_class_rgb(mask), transforms.ToTensor()(image1), transforms.ToTensor()(image2),  \
            self.mask_to_class_rgb(mask1), self.mask_to_class_rgb(mask2)
        else:
            # if self.dataset == 'idrid':
            #     return transforms.ToTensor()(A.resize(image, 576, 576, interpolation=cv2.INTER_AREA)), A.resize(self.mask_to_class_rgb(mask), 576, 576, interpolation=cv2.INTER_AREA)
            # else:
            if self.augmentation is not None:
                transformed = self.augmentation(image=image, mask=mask)
                image, mask = transformed['image'], transformed['mask']
                return transforms.ToTensor()(image), self.mask_to_class_rgb(mask)
            else:
                return transforms.ToTensor()(image), self.mask_to_class_rgb(mask)

