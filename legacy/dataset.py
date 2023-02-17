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

class BratsTrainDataset(Dataset):

    def __init__(self, image_path, mask_path, augmentation=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_path) 

    def __getitem__(self, index):
        image_path = self.image_path[index]
        mask_path = self.mask_path[index]
        # unlabeled_image_path = self.unlabeled_image_path[index]

        image = np.load(image_path).astype('float32')
        mask = np.load(mask_path).astype('uint8')
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
        
        if self.augmentation is not None:
            image1, whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1 = self.data_transform(
                image, whole_tumor_label, tumor_core_label, enhanced_tumor_label
            )
            image2, whole_tumor_label2, tumor_core_label2, enhanced_tumor_label2 = self.data_transform(
                image, whole_tumor_label, tumor_core_label, enhanced_tumor_label
            )

        label1 = np.concatenate((whole_tumor_label1, tumor_core_label1, enhanced_tumor_label1), axis=0)
        label2 = np.concatenate((whole_tumor_label2, tumor_core_label2, enhanced_tumor_label2), axis=0)
        return image1.astype('float32'), image2.astype('float32'), label1.astype('uint8'), label2.astype('uint8')

    def data_transform(self, image, whole_tumor_label, tumor_core_label, enhanced_tumor_label):
        transformed = self.augmentation(image=image[0, :, :], t1=image[1, :, :], t1ce=image[2, :, :], t2=image[3, :, :], mask=whole_tumor_label, tumorCore=tumor_core_label, enhancingTumor=enhanced_tumor_label)
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



class BratsValidationDataset(Dataset):

    def __init__(self, image_path, mask_path, augmentation=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_path) 

    def __getitem__(self, index):
        image_path = self.image_path[index]
        mask_path = self.mask_path[index]

        image = np.load(image_path)
        mask = np.load(mask_path)
        
        whole_tumor_label = mask.copy()
        whole_tumor_label[mask==1] = 1
        whole_tumor_label[mask==2] = 1
        whole_tumor_label[mask==3] = 1
        whole_tumor_label = np.expand_dims(whole_tumor_label, axis=0)

        tumor_core_label = mask.copy()
        tumor_core_label[mask==1] = 1
        tumor_core_label[mask==2] = 0
        tumor_core_label[mask==3] = 1
        tumor_core_label = np.expand_dims(tumor_core_label, axis=0)

        enhanced_tumor_label = mask.copy()
        enhanced_tumor_label[mask==1] = 0
        enhanced_tumor_label[mask==2] = 0
        enhanced_tumor_label[mask==3] = 1
        enhanced_tumor_label = np.expand_dims(enhanced_tumor_label, axis=0)

        label = np.concatenate((whole_tumor_label, tumor_core_label, enhanced_tumor_label), axis=0)
        
        return image.astype('float32'), label.astype('uint8')





class BratsSupervisedDataset(Dataset):

    def __init__(self, image_path, mask_path, augmentation=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_path) 

    def __getitem__(self, index):
        image_path = self.image_path[index]
        mask_path = self.mask_path[index]

        image = np.load(image_path).astype('float32')
        mask = np.load(mask_path).astype('uint8')
        
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

        if self.augmentation is not None:
            image, whole_tumor_label, tumor_core_label, enhanced_tumor_label = self.data_transform(
                image, whole_tumor_label, tumor_core_label, enhanced_tumor_label
            )

        label = np.concatenate((whole_tumor_label, tumor_core_label, enhanced_tumor_label), axis=0)

        return image.astype('float32'), label.astype('uint8')

    def data_transform(self, image, whole_tumor_label, tumor_core_label, enhanced_tumor_label):
        transformed = self.augmentation(image=image[0, :, :], t1=image[1, :, :], t1ce=image[2, :, :], t2=image[3, :, :], mask=whole_tumor_label, tumorCore=tumor_core_label, enhancingTumor=enhanced_tumor_label)
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


class BratsUnsupervisedDataset(Dataset):

    def __init__(self, image_path, augmentation=None):
        self.image_path = image_path
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_path) 

    def __getitem__(self, index):
        image_path = self.image_path[index]

        image = np.load(image_path).astype('float32')
        
        image1 = self.data_transform(image)
        image2 = self.data_transform(image)

        return image1.astype('float32'), image2.astype('float32')

    def data_transform(self, image):
        transformed = self.augmentation(image=image[0, :, :], t1=image[1, :, :], t1ce=image[2, :, :], t2=image[3, :, :])
        flair = np.expand_dims(transformed["image"], axis=0)
        t1 = np.expand_dims(transformed['t1'], axis=0)
        t1ce = np.expand_dims(transformed['t1ce'], axis=0)
        t2 = np.expand_dims(transformed['t2'], axis=0)
        image = np.concatenate((flair, t1, t1ce, t2), axis=0)

        return image

class ConstrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder
    
    @staticmethod    
    def get_simclr_pipeline_transform(size, s=1):
        data_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomAffine(degrees=0, translate=(0, 0.25)),
                                            transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
                                            transforms.RandomResizedCrop(size=size),
                                            transforms.GaussianBlur(),
                                            transforms.Grayscale(),
                                            transforms.ToTensor()])
        return data_transform

    def get_dataset(self, name, n_views):
        dataset = BratsTrainDataset
