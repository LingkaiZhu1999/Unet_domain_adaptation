# File: dataset.py

import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import cv2  # Used for resizing, install with: pip install opencv-python

class NiftiDataset(Dataset):
    """
    PyTorch Dataset for loading FULL 3D NIfTI volumes.
    Best for inference or validation on the entire volume.
    """
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        self.has_labels = self.hf_dataset[0]['label_path'] != "N/A"

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image_itk = sitk.ReadImage(item['image_path'], sitk.sitkFloat32)
        image_np = sitk.GetArrayFromImage(image_itk)
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)

        if self.has_labels:
            label_itk = sitk.ReadImage(item['label_path'], sitk.sitkUInt8)
            label_np = sitk.GetArrayFromImage(label_itk)
            label_tensor = torch.from_numpy(label_np).long()
            return {"image": image_tensor, "label": label_tensor}
        else:
            return {"image": image_tensor, "label": torch.tensor(-1)}


class FlareSliceDataset(Dataset):
    """
    Dataset for the FLARE challenge, similar in style to the BraTS dataset.

    - Loads data from the Hugging Face dataset object.
    - Extracts a single 2D slice from the 3D volume.
    - Intelligently selects slices that contain labels (if available).
    - Applies normalization and resizing.
    - Can generate two augmented views for contrastive learning.
    """
    def __init__(self, hf_dataset, augmentation=None, output_size=(512, 512), is_contrastive=False):
        """
        Args:
            hf_dataset: The Hugging Face dataset object from load_dataset.
            augmentation: An augmentation pipeline (e.g., from albumentations).
            output_size (tuple): The desired output size for each slice (H, W).
            is_contrastive (bool): If True, __getitem__ returns two augmented views.
                                   If False, returns a single view.
        """
        self.hf_dataset = hf_dataset
        self.augmentation = augmentation
        self.output_size = output_size
        self.is_contrastive = is_contrastive
        self.has_labels = self.hf_dataset[0]['label_path'] != "N/A"

    def __len__(self):
        return len(self.hf_dataset)

    def _normalize(self, image):
        """ Z-score normalization based on non-zero pixels. """
        pixels = image[image > 0]
        if pixels.size == 0:
            return image # Avoid division by zero
        mean = pixels.mean()
        std = pixels.std()
        if std == 0:
            return (image - mean)
        return (image - mean) / std

    # def _resize_slice(self, image, label):
    #     """ Resizes a 2D slice and its corresponding label. """
    #     # Use INTER_LINEAR for the image and INTER_NEAREST for the label mask
    #     resized_image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LINEAR)
    #     resized_label = cv2.resize(label, self.output_size, interpolation=cv2.INTER_NEAREST)
    #     return resized_image, resized_label

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Load the full 3D volumes
        image_3d_itk = sitk.ReadImage(item['image_path'], sitk.sitkFloat32)
        image_3d = sitk.GetArrayFromImage(image_3d_itk)
        
        if self.has_labels:
            label_3d_itk = sitk.ReadImage(item['label_path'], sitk.sitkUInt8)
            label_3d = sitk.GetArrayFromImage(label_3d_itk)
        else:
            # Create a dummy label volume if none exists
            label_3d = np.zeros_like(image_3d, dtype=np.uint8)

        # --- Smart Slice Selection (similar to your BraTS 'get_slice') ---
        # Find all slices that contain a label
        z_indices_with_labels = np.where(np.any(label_3d, axis=(1, 2)))[0]
        
        if len(z_indices_with_labels) > 0:
            # If there are labeled slices, pick one randomly
            slice_idx = np.random.choice(z_indices_with_labels)
        else:
            # Otherwise, just pick a random slice from the middle half of the volume
            start = image_3d.shape[0] // 4
            end = start * 3
            slice_idx = np.random.randint(start, end) if start < end else image_3d.shape[0] // 2
            
        image_2d = image_3d[slice_idx, :, :]
        label_2d = label_3d[slice_idx, :, :]

        # --- Preprocessing (Normalization and Resizing) ---
        image_2d = self._normalize(image_2d)
        # resize if output_size is specified
        # if self.output_size is not None:
        #     # Resize the 2D slice and its label
        #     # image_2d, label_2d = self._resize_slice(image_2d, label_2d)
        #     # center crop the image and label to output_size
        h, w = image_2d.shape
        target_h, target_w = self.output_size
        if h > target_h or w > target_w:
            # Center crop the image and label to the target size
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
            image_2d = image_2d[start_h:start_h + target_h, start_w:start_w + target_w]
            label_2d = label_2d[start_h:start_h + target_h, start_w:start_w + target_w]

        # --- Data Augmentation ---
        if self.augmentation is not None:
            # Create the first augmented view
            transformed1 = self.augmentation(image=image_2d, mask=label_2d)
            image1 = transformed1['image']
            label1 = transformed1['mask']
            
            if self.is_contrastive:
                # Create a second, different augmented view
                transformed2 = self.augmentation(image=image_2d, mask=label_2d)
                image2 = transformed2['image']
                label2 = transformed2['mask']
            
        else:
            # If no augmentation, just use the preprocessed slice
            image1 = image_2d
            label1 = label_2d
            if self.is_contrastive:
                image2 = image_2d
                label2 = label_2d

        # --- Final Formatting to PyTorch Tensors ---
        # Add channel dimension to images and ensure correct types
        image1 = torch.from_numpy(image1).float().unsqueeze(0)
        label1 = torch.from_numpy(label1).long()
        
        if self.is_contrastive:
            image2 = torch.from_numpy(image2).float().unsqueeze(0)
            label2 = torch.from_numpy(label2).long()
            return image1, image2, label1, label2
        else:
            # For standard supervised learning, return a dictionary
            return {"image": image1, "label": label1}