import numpy as np
import cv2
import random

from skimage.io import imread
from skimage import color
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.transforms.functional import gaussian_blur, affine
import albumentations as A
import os
import nibabel as nib
from random import randint
from monai import transforms

class BratsTrainContrastDataset(Dataset):
    def __init__(self, datapath='/mnt/asgard2/data/lingkai/braTS20/BraTS2020_TrainingData', augmentation=None):
        self.augmentaion = augmentation
        self.datapath = datapath
        self.volumeType = ['flair', 't1ce', 't1', 't2', 'seg']
    def __getitem__(self, index):
        # 1st volume
        images = {}
        folderpaths = os.path.join(self.datapath, f'BraTS20_Training_{str(index+1).zfill(3)}')
        for name in self.volumeType:
            img = nib.load(os.path.join(folderpaths, f'BraTS20_Training_{str(index+1).zfill(3)}_{name}.nii')).get_fdata()
            if name == 'seg':
                img[img==4] = 3
            images[name] = img
        # normalize the non-zero voxels in images
        images['flair'] = self.normalize(images['flair'])
        images['t1ce'] = self.normalize(images['t1ce'])
        images['t1'] = self.normalize(images['t1'])
        images['t2'] = self.normalize(images['t2'])

        image_slice1, seg_slice1 = self.get_slice(images)
        image_slice2, seg_slice2 = self.get_slice(images)
        whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1 = self.mask_label_process(seg_slice1)
        whole_tumor_label2, tumor_core_label2, enhanced_tumor_label2 = self.mask_label_process(seg_slice2)
        if self.augmentaion is not None:
            image1, whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1 = self.data_transform(
                image_slice1, whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1
            )
            image2, whole_tumor_label2, tumor_core_label2, enhanced_tumor_label2 = self.data_transform(
                image_slice2, whole_tumor_label2, tumor_core_label2, enhanced_tumor_label2
            )
        label1 = np.concatenate((whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1), axis=0)
        label2 = np.concatenate((whole_tumor_label2, tumor_core_label2, enhanced_tumor_label2), axis=0)
        return image1, image2, label1, label2

    def __len__(self):
        return len(os.listdir(self.datapath)) - 1
    
    def crop_center(self, img, cropx=160, cropy=160):
        y, x = img.shape
        startx = x//2 - cropx//2
        starty = y//2 - cropy//2    
        return img[starty:starty+cropy, startx:startx+cropx]


    def normalize(self, input):
        normalizeIntensity = transforms.NormalizeIntensity(nonzero=True)
        input_norm = normalizeIntensity(input)
        return input_norm

    def get_slice(self, images):
        _, _, max_z = images['flair'].shape
        while True:
            slice_z_num = randint(0, max_z-1)
            if np.max(images['seg'][:, :, slice_z_num]) != 0: break

        flair = np.expand_dims(self.crop_center(images['flair'][:, :, slice_z_num]),axis=0)
        t1ce = np.expand_dims(self.crop_center(images['t1ce'][:, :, slice_z_num]), axis=0)
        t1 = np.expand_dims(self.crop_center(images['t1'][:, :, slice_z_num]), axis=0)
        t2 = np.expand_dims(self.crop_center(images['t2'][:, :, slice_z_num]), axis=0)
        seg = self.crop_center(images['seg'][:, :, slice_z_num])
        image = np.concatenate((flair, t1ce, t1, t2), axis=0)
        # 2nd 
        return image.astype('float32'), seg.astype('uint8')

    def data_transform(self, image, whole_tumor_label, tumor_core_label, enhanced_tumor_label):
        transformed = self.augmentaion(image=image[0, :, :], t1=image[1, :, :], t1ce=image[2, :, :], t2=image[3, :, :], mask=whole_tumor_label, tumorCore=tumor_core_label, enhancingTumor=enhanced_tumor_label)
        flair = np.expand_dims(transformed["image"], axis=0)
        t1 = np.expand_dims(transformed['t1'], axis=0)
        t1ce = np.expand_dims(transformed['t1ce'], axis=0)
        t2 = np.expand_dims(transformed['t2'], axis=0)
        image = np.concatenate((flair, t1, t1ce, t2), axis=0)

        whole_tumor_label = transformed["mask"] # [w, h]
        tumor_core_label = transformed['tumorCore']
        enhanced_tumor_label = transformed['enhancingTumor']
        whole_tumor_label = np.expand_dims(whole_tumor_label, axis=0) #[1, w, h]
        tumor_core_label = np.expand_dims(tumor_core_label, axis=0)
        enhanced_tumor_label = np.expand_dims(enhanced_tumor_label, axis=0)
        return image, whole_tumor_label, tumor_core_label, enhanced_tumor_label

    def mask_label_process(self, mask):
        whole_tumor_label = mask.copy()
        whole_tumor_label[mask==1] = 1
        whole_tumor_label[mask==2] = 1
        whole_tumor_label[mask==3] = 1
        

        tumor_core_label = mask.copy()
        tumor_core_label[mask==1] = 1
        tumor_core_label[mask==2] = 0
        tumor_core_label[mask==3] = 1
        

        enhanced_tumor_label = mask.copy()
        enhanced_tumor_label[mask==1] = 0
        enhanced_tumor_label[mask==2] = 0
        enhanced_tumor_label[mask==3] = 1

        return whole_tumor_label, tumor_core_label, enhanced_tumor_label

if __name__ == "__main__":
    train_transform = A.Compose([
        # A.Resize(200, 200),
        # A.CropNonEmptyMaskIfExists(height=150, width=150),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(1.0, 1.5), p=0.15),
        A.Affine(translate_percent=(0, 0.25), p=0.5),
        A.GaussianBlur(sigma_limit=(0.5, 1.5), p=0.15), 
        A.GaussNoise(var_limit=(0, 0.33), p=0.15),
        A.RandomBrightness(limit=(0.7, 1.3), p=0.15),
        A.RandomContrast(limit=(0.65, 1.5), p=0.15)],
        additional_targets={'t1': 'image', 't1ce': 'image', 't2': 'image', 'tumorCore': 'mask', 'enhancingTumor': 'mask'}
        )

    train_dataset = BratsTrainContrastDataset(augmentation=train_transform)
    dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True,pin_memory=True,drop_last=False)
    for batch_idx, data in enumerate(dataloader):
        image1, image2, label1, label2 = data