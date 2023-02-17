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
    """
    Dataset for contrastive learning of the BraTS 2020 challenge dataset.
    """
    def __init__(self, datapath='/mnt/asgard2/data/lingkai/braTS20/BraTS2020_TrainingData', augmentation=None):
        self.augmentaion = augmentation
        self.datapath = datapath
        self.volumeType = ['flair', 't1ce', 't1', 't2', 'seg']
        self.IsCrop = True
    def __getitem__(self, index):
        # 1st volume
        images = {}
        folderpath = self.datapath[index]
        for name in self.volumeType:
            img = np.abs(nib.load(os.path.join(folderpath, f'{folderpath[-20:]}_{name}.nii')).get_fdata().astype(np.int16))
            # img = np.abs(nib.load(os.path.join(folderpath, f'{folderpath[40:]}_{name}.nii')).get_fdata().astype(np.int16))
            img[img == -32768] = 0
            if name == 'seg':
                img[img==4] = 3
            images[name] = img
        # normalize the non-zero voxels in images
        images['flair'] = self.normalize(images['flair'])
        images['t1ce'] = self.normalize(images['t1ce'])
        images['t1'] = self.normalize(images['t1'])
        images['t2'] = self.normalize(images['t2'])

        image_slice1, image_slice2, seg_slice1, seg_slice2 = self.get_slice(images)
        whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1 = self.mask_label_process(seg_slice1)
        whole_tumor_label2, tumor_core_label2, enhanced_tumor_label2 = self.mask_label_process(seg_slice2)
        if self.augmentaion is not None:
            image1, whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1 = self.data_transform(
                image_slice1, whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1
            )
            image2, whole_tumor_label2, tumor_core_label2, enhanced_tumor_label2 = self.data_transform(
                image_slice2, whole_tumor_label2, tumor_core_label2, enhanced_tumor_label2
            )
        else: 
            image1 = image_slice1
            image2 = image_slice2
            whole_tumor_label1 = np.expand_dims(whole_tumor_label1, axis=0) #[1, w, h]
            tumor_core_label1 = np.expand_dims(tumor_core_label1, axis=0)
            enhanced_tumor_label1 = np.expand_dims(enhanced_tumor_label1, axis=0)

            whole_tumor_label2 = np.expand_dims(whole_tumor_label2, axis=0) #[1, w, h]
            tumor_core_label2 = np.expand_dims(tumor_core_label2, axis=0)
            enhanced_tumor_label2 = np.expand_dims(enhanced_tumor_label2, axis=0)
        label1 = np.concatenate((whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1), axis=0)
        label2 = np.concatenate((whole_tumor_label2, tumor_core_label2, enhanced_tumor_label2), axis=0)
        return image1, image2, label1, label2

    def __len__(self):
        return len(self.datapath) - 1
    
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
        if self.IsCrop:
            flair = np.expand_dims(self.crop_center(images['flair'][:, :, slice_z_num]),axis=0)
            t1ce = np.expand_dims(self.crop_center(images['t1ce'][:, :, slice_z_num]), axis=0)
            t1 = np.expand_dims(self.crop_center(images['t1'][:, :, slice_z_num]), axis=0)
            t2 = np.expand_dims(self.crop_center(images['t2'][:, :, slice_z_num]), axis=0)
            seg = self.crop_center(images['seg'][:, :, slice_z_num])
        else:
            flair = np.expand_dims(images['flair'][:, :, slice_z_num],axis=0)
            t1ce = np.expand_dims(images['t1ce'][:, :, slice_z_num], axis=0)
            t1 = np.expand_dims(images['t1'][:, :, slice_z_num], axis=0)
            t2 = np.expand_dims(images['t2'][:, :, slice_z_num], axis=0)
            seg = images['seg'][:, :, slice_z_num]
        # 2nd
        while True:
            if slice_z_num - 0 < 5:
                left_bound = 0
            else:
                left_bound = slice_z_num - 5
            if slice_z_num + 5 > max_z:
                right_bound = max_z
            else:
                right_bound = slice_z_num + 5 
            slice_z_num1 = randint(left_bound, right_bound)
            if np.max(images['seg'][:, :, slice_z_num1]) != 0: break
            # if index1 != index: break
        if self.IsCrop:
            flair_1 = np.expand_dims(self.crop_center(images['flair'][:, :, slice_z_num1]),axis=0)
            t1ce_1 = np.expand_dims(self.crop_center(images['t1ce'][:, :, slice_z_num1]), axis=0)
            t1_1 = np.expand_dims(self.crop_center(images['t1'][:, :, slice_z_num1]), axis=0)
            t2_1 = np.expand_dims(self.crop_center(images['t2'][:, :, slice_z_num1]), axis=0)
            seg_1 = self.crop_center(images['seg'][:, :, slice_z_num1])
        else:
            flair_1 = np.expand_dims(images['flair'][:, :, slice_z_num1],axis=0)
            t1ce_1 = np.expand_dims(images['t1ce'][:, :, slice_z_num1], axis=0)
            t1_1 = np.expand_dims(images['t1'][:, :, slice_z_num1], axis=0)
            t2_1 = np.expand_dims(images['t2'][:, :, slice_z_num1], axis=0)
            seg_1 = images['seg'][:, :, slice_z_num1]
        image = np.concatenate((flair, t1, t1ce, t2), axis=0)
        image1 = np.concatenate((flair_1, t1_1, t1ce_1, t2_1), axis=0)
        return image.astype('float32'), image1.astype('float32'), seg.astype('uint8'), seg_1.astype('uint8')

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

class BratsTrainContrastDataset_only_data_aug(Dataset):
    """
    Dataset for contrastive learning of the BraTS 2020 challenge dataset.
    Only data augmentation is implemented, while pair generation is not.
    """
    def __init__(self, datapath='/mnt/asgard2/data/lingkai/braTS20/BraTS2020_TrainingData', augmentation=None):
        self.augmentaion = augmentation
        self.datapath = datapath
        self.volumeType = ['flair', 't1ce', 't1', 't2', 'seg']
        self.IsCrop = True

    def __getitem__(self, index):
        # 1st volume
        images = {}
        folderpath = self.datapath[index]
        for name in self.volumeType:
            img = np.abs(nib.load(os.path.join(folderpath, f'{folderpath[-20:]}_{name}.nii')).get_fdata().astype(np.int16))
            # img = np.abs(nib.load(os.path.join(folderpath, f'{folderpath[40:]}_{name}.nii')).get_fdata().astype(np.int16))
            img[img == -32768] = 0
            if name == 'seg':
                img[img==4] = 3
            images[name] = img
        # normalize the non-zero voxels in images
        images['flair'] = self.normalize(images['flair'])
        images['t1ce'] = self.normalize(images['t1ce'])
        images['t1'] = self.normalize(images['t1'])
        images['t2'] = self.normalize(images['t2'])

        image_slice, seg_slice = self.get_slice(images)
        whole_tumor_label, tumor_core_label, enhanced_tumor_label = self.mask_label_process(seg_slice)
        if self.augmentaion is not None:
            image1, whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1 = self.data_transform(
                image_slice, whole_tumor_label, tumor_core_label, enhanced_tumor_label
            )
            image2, whole_tumor_label2, tumor_core_label2, enhanced_tumor_label2 = self.data_transform(
                image_slice, whole_tumor_label, tumor_core_label, enhanced_tumor_label
            )
        else: 
            image1 = image_slice
            image2 = image_slice
            whole_tumor_label1 = np.expand_dims(whole_tumor_label1, axis=0) #[1, w, h]
            tumor_core_label1 = np.expand_dims(tumor_core_label1, axis=0)
            enhanced_tumor_label1 = np.expand_dims(enhanced_tumor_label1, axis=0)

            whole_tumor_label2 = np.expand_dims(whole_tumor_label2, axis=0) #[1, w, h]
            tumor_core_label2 = np.expand_dims(tumor_core_label2, axis=0)
            enhanced_tumor_label2 = np.expand_dims(enhanced_tumor_label2, axis=0)
        label1 = np.concatenate((whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1), axis=0)
        label2 = np.concatenate((whole_tumor_label2, tumor_core_label2, enhanced_tumor_label2), axis=0)
        return image1, image2, label1, label2

    def __len__(self):
        return len(self.datapath) - 1
    
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
        if self.IsCrop:
            flair = np.expand_dims(self.crop_center(images['flair'][:, :, slice_z_num]),axis=0)
            t1ce = np.expand_dims(self.crop_center(images['t1ce'][:, :, slice_z_num]), axis=0)
            t1 = np.expand_dims(self.crop_center(images['t1'][:, :, slice_z_num]), axis=0)
            t2 = np.expand_dims(self.crop_center(images['t2'][:, :, slice_z_num]), axis=0)
            seg = self.crop_center(images['seg'][:, :, slice_z_num])
        else:
            flair = np.expand_dims(images['flair'][:, :, slice_z_num],axis=0)
            t1ce = np.expand_dims(images['t1ce'][:, :, slice_z_num], axis=0)
            t1 = np.expand_dims(images['t1'][:, :, slice_z_num], axis=0)
            t2 = np.expand_dims(images['t2'][:, :, slice_z_num], axis=0)
            seg = images['seg'][:, :, slice_z_num]
        image = np.concatenate((flair, t1, t1ce, t2), axis=0)
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

class BratsSuperviseTrainDataset(Dataset):
    def __init__(self, datapaths, augmentation=None, IsCrop=True, IsSlice=True):
        self.augmentaion = augmentation
        self.datapath = datapaths
        self.volumeType = ['flair', 't1ce', 't1', 't2', 'seg']
        self.IsCrop = IsCrop
        self.IsSlice = IsSlice
    
    def __getitem__(self, index):
        images = {}
        # folderpaths = os.path.join(self.datapath, f'BraTS20_Training_{str(index).zfill(3)}')
        folderpath = self.datapath[index]
        for name in self.volumeType:
            img = np.abs(nib.load(os.path.join(folderpath, f'{folderpath[-20:]}_{name}.nii')).get_fdata().astype(np.int16))
            # img = np.abs(nib.load(os.path.join(folderpath, f'{folderpath[40:]}_{name}.nii')).get_fdata().astype(np.int16))
            img[img == -32768] = 0
            if name == 'seg':
                img[img==4] = 3
            images[name] = img
        # normalize the non-zero voxels in images
        images['flair'] = self.normalize(images['flair'])
        images['t1ce'] = self.normalize(images['t1ce'])
        images['t1'] = self.normalize(images['t1'])
        images['t2'] = self.normalize(images['t2'])
        if self.IsSlice == True:
            image1, seg_slice1 = self.get_slice(images)
            whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1 = self.mask_label_process(seg_slice1)
            if self.augmentaion is not None:
                image1, whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1 = self.data_transform(
                    image1, whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1
                )
                label1 = np.concatenate((whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1), axis=0)
            else:
                label1 = np.concatenate((np.expand_dims(whole_tumor_label1, 0), np.expand_dims(tumor_core_label1, 0), np.expand_dims(enhanced_tumor_label1, 0)), axis=0)
            return image1, label1
        else:
            whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1 = self.mask_label_process(images['seg'])
        
            label = np.concatenate((np.expand_dims(whole_tumor_label1, 0), np.expand_dims(tumor_core_label1, 0), np.expand_dims(enhanced_tumor_label1, 0)), axis=0)
            return images, label


    def __len__(self):
        return len(self.datapath) - 1
    
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
        if self.IsCrop:
            flair = np.expand_dims(self.crop_center(images['flair'][:, :, slice_z_num]),axis=0)
            t1ce = np.expand_dims(self.crop_center(images['t1ce'][:, :, slice_z_num]), axis=0)
            t1 = np.expand_dims(self.crop_center(images['t1'][:, :, slice_z_num]), axis=0)
            t2 = np.expand_dims(self.crop_center(images['t2'][:, :, slice_z_num]), axis=0)
            seg = self.crop_center(images['seg'][:, :, slice_z_num])
        else:
            flair = np.expand_dims(images['flair'][:, :, slice_z_num],axis=0)
            t1ce = np.expand_dims(images['t1ce'][:, :, slice_z_num], axis=0)
            t1 = np.expand_dims(images['t1'][:, :, slice_z_num], axis=0)
            t2 = np.expand_dims(images['t2'][:, :, slice_z_num], axis=0)
            seg = images['seg'][:, :, slice_z_num]
        image = np.concatenate((flair, t1, t1ce, t2), axis=0)

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

class BratsTestDataset(Dataset):
    def __init__(self, datapaths, augmentation=None):
        self.augmentaion = augmentation
        self.datapath = datapaths
        self.volumeType = ['flair', 't1ce', 't1', 't2', 'seg']
    
    def __getitem__(self, index):
        images = {}
        # folderpaths = os.path.join(self.datapath, f'BraTS20_Training_{str(index).zfill(3)}')
        folderpath = self.datapath[index]
        for name in self.volumeType:
            img = np.abs(nib.load(os.path.join(folderpath, f'{folderpath[-20:]}_{name}.nii')).get_fdata().astype(np.int16))
            # img = np.abs(nib.load(os.path.join(folderpath, f'{folderpath[40:]}_{name}.nii')).get_fdata().astype(np.int16))
            img[img == -32768] = 0
            if name == 'seg':
                img[img==4] = 3
            images[name] = img
        # normalize the non-zero voxels in images
        images['flair'] = self.normalize(images['flair'])
        images['t1ce'] = self.normalize(images['t1ce'])
        images['t1'] = self.normalize(images['t1'])
        images['t2'] = self.normalize(images['t2'])

        whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1 = self.mask_label_process(images['seg'])
        
        label = np.concatenate((np.expand_dims(whole_tumor_label1, 0), np.expand_dims(tumor_core_label1, 0), np.expand_dims(enhanced_tumor_label1, 0)), axis=0)
        return images, label

    def __len__(self):
        return len(self.datapath) - 1
    
    def normalize(self, input):
        normalizeIntensity = transforms.NormalizeIntensity(nonzero=True)
        input_norm = normalizeIntensity(input)
        return input_norm


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