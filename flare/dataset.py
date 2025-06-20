# File: dataset.py (New and Improved Version)

import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import albumentations as A


# NiftiDataset class remains the same...
class NiftiDataset(Dataset):
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
    Robust Dataset for the FLARE challenge.
    Handles variable input sizes optimally by resizing with aspect ratio preservation.
    """
    def __init__(self, hf_dataset, augmentation=None, output_size=(480, 480), is_contrastive=False):
        self.hf_dataset = hf_dataset
        self.augmentation = augmentation # The main augmentation pipeline
        self.output_size = output_size
        self.is_contrastive = is_contrastive
        self.has_labels = self.hf_dataset[0]['label_path'] != "N/A"
        
        # --- THIS IS THE NEW, OPTIMAL PREPROCESSING PIPELINE ---
        # It handles all cases: images smaller, larger, or non-square.
        self.preprocessing = A.Compose([
            # Step 1: Resize the longest side of the image to output_size, maintaining aspect ratio.
            # If image is 1024x512 and output_size is 480, it becomes 480x240.
            # If image is 200x100 and output_size is 480, it becomes 480x240.
            A.LongestMaxSize(max_size=max(output_size), interpolation=1), # 1=cv2.INTER_LINEAR
            
            # Step 2: Pad the image to be exactly output_size.
            # The padding is added symmetrically to the borders.
            A.PadIfNeeded(
                min_height=output_size[0],
                min_width=output_size[1],
                border_mode=0, # 0=cv2.BORDER_CONSTANT
            )
        ])
        # --------------------------------------------------------

    def __len__(self):
        return len(self.hf_dataset)

    def _normalize(self, image):
        # Clip to 3 standard deviations and then scale to [0, 1] for stability
        # This is a common and robust normalization for medical images.
        min_val = -350
        max_val = 350
        clipped_image = np.clip(image, min_val, max_val)
        
        # Scale to 0-1 range
        return (clipped_image - min_val) / (max_val - min_val)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image_3d = sitk.GetArrayFromImage(sitk.ReadImage(item['image_path'], sitk.sitkFloat32))
        
        if self.has_labels:
            label_3d = sitk.GetArrayFromImage(sitk.ReadImage(item['label_path'], sitk.sitkUInt8))
        else:
            label_3d = np.zeros_like(image_3d, dtype=np.uint8)

        # Smart Slice Selection (remains the same)
        z_indices_with_labels = np.where(np.any(label_3d, axis=(1, 2)))[0]
        if len(z_indices_with_labels) > 0:
            slice_idx = np.random.choice(z_indices_with_labels)
        else:
            start = image_3d.shape[0] // 4
            end = start * 3
            slice_idx = np.random.randint(start, end) if start < end else image_3d.shape[0] // 2
            
        image_2d = image_3d[slice_idx, :, :]
        label_2d = label_3d[slice_idx, :, :]

        # --- NEW PREPROCESSING WORKFLOW ---
        
        # 1. Apply robust normalization
        image_2d = self._normalize(image_2d)

        # 2. Apply the resize-and-pad preprocessing
        preprocessed = self.preprocessing(image=image_2d, mask=label_2d)
        image_prep = preprocessed['image']
        label_prep = preprocessed['mask']

        # 3. Apply the main data augmentations (if any)
        if self.augmentation is not None:
            transformed1 = self.augmentation(image=image_prep, mask=label_prep)
            image1 = transformed1['image']
            label1 = transformed1['mask']
            
            if self.is_contrastive:
                transformed2 = self.augmentation(image=image_prep, mask=label_prep)
                image2 = transformed2['image']
                label2 = transformed2['mask']
        else:
            image1 = image_prep
            label1 = label_prep
            if self.is_contrastive:
                image2 = image_prep
                label2 = label_prep
        
        # Final formatting to PyTorch Tensors
        image1 = torch.from_numpy(image1).float().unsqueeze(0)
        label1 = self.mask_label_process(torch.from_numpy(label1))

        if self.is_contrastive:
            image2 = torch.from_numpy(image2).float().unsqueeze(0)
            label2 = self.mask_label_process(torch.from_numpy(label2))
            return image1, image2, label1, label2
        else:
            return {"image": image1, "label": label1}
        
    def mask_label_process(self, mask):
        """ Process the mask labels to ensure they are in the correct format.{
        "background": 0,
        "liver": 1,
        "right-kidney": 2,
        "spleen": 3,
        "pancreas": 4,
        "aorta": 5,
        "ivc": 6,
        "rag": 7,
        "lag": 8,
        "gallbladder": 9,
        "esophagus": 10,
        "stomach": 11,
        "duodenum": 12,
        "left kidney": 13
    },
        """
        background_mask = (mask == 0)
        liver_mask = (mask == 1)
        right_kidney_mask = (mask == 2)
        spleen_mask = (mask == 3)
        pancreas_mask = (mask == 4)
        aorta_mask = (mask == 5)
        ivc_mask = (mask == 6)
        rag_mask = (mask == 7)
        lag_mask = (mask == 8)
        gallbladder_mask = (mask == 9)
        esophagus_mask = (mask == 10)
        stomach_mask = (mask == 11)
        duodenum_mask = (mask == 12)
        left_kidney_mask = (mask == 13)
        # concat all masks into a single tensor
        processed_mask = torch.stack([
            background_mask,
            liver_mask,
            right_kidney_mask,
            spleen_mask,
            pancreas_mask,
            aorta_mask,
            ivc_mask,
            rag_mask,
            lag_mask,
            gallbladder_mask,
            esophagus_mask,
            stomach_mask,
            duodenum_mask,
            left_kidney_mask
        ], dim=0).float()

        return processed_mask
