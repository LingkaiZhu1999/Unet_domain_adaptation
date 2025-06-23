import torch
# from torch.utils.data import Dataset
from monai.data import Dataset
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore", ".*unexpected scales in sform.*")
# Import the necessary 3D-capable MONAI transforms
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandAffined,
    RandSpatialCropd,
    MapTransform,
    Pad,
    ResizeWithPadOrCropd,
    RandRotate90d,
)

from monai.data import NibabelReader
import albumentations as A
from monai.transforms import Lambdad
import torch
import random
from typing import Dict, Union, List, Any
from datasets import load_dataset
import numpy as np
# --- STEP 1: Create a simple, self-contained custom transform ---

NUM_CLASSES = 14  # Number of classes in the FLARE-3D dataset

# In dataset.py, near the top
class SafeNormalizeIntensityd(MapTransform):
    """
    A wrapper for NormalizeIntensityd that adds a small epsilon to the
    standard deviation to prevent division-by-zero errors when an image
    patch is all zeros.
    """
    def __init__(self, keys: list, nonzero: bool = True, channel_wise: bool = True, subtrahend=None, divisor=None):
        super().__init__(keys, allow_missing_keys=True)
        self.normalizer = NormalizeIntensityd(
            keys=keys, 
            nonzero=nonzero, 
            channel_wise=channel_wise,
            subtrahend=subtrahend,
            divisor=divisor
        )
        self.nonzero = nonzero

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            if key in d:
                img = d[key]
                # If nonzero is True, check if there are any non-zero pixels
                if self.nonzero and torch.count_nonzero(img) == 0:
                    # If the image is all zeros, normalization is not needed and unsafe.
                    # The image remains all zeros.
                    continue
                
                # For non-zero images, apply the standard normalizer
                # We can add an epsilon to the divisor calculation within the original,
                # but a simpler safe-guard is to just check std dev.
                pix_to_consider = img[img != 0] if self.nonzero else img.flatten()
                if pix_to_consider.numel() > 0 and torch.std(pix_to_consider) > 1e-6:
                    d = self.normalizer(d)
                # else: if std is zero, do nothing, the image is constant.
        return d
    
class ClipIntensityPercentiled(MapTransform):
    """
    Dictionary-based transform to clip intensity values based on calculated percentiles.
    This is the standard approach for MRI normalization.
    """
    def __init__(self, keys: list, lower_percentile: float, upper_percentile: float):
        """
        Args:
            keys: Keys of the tensors to clip.
            lower_percentile: The lower percentile (e.g., 0.5).
            upper_percentile: The upper percentile (e.g., 99.5).
        """
        super().__init__(keys, allow_missing_keys=True)
        self.lower_p = lower_percentile
        self.upper_p = upper_percentile

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            if key in d:
                img_tensor = d[key]
                
                # Calculate the percentile values from the non-zero elements of the image
                # This is important to avoid the large number of background zeros skewing the percentiles
                non_zero_vals = img_tensor[img_tensor > 0]
                if non_zero_vals.numel() == 0:
                    # If image is all zeros, do nothing
                    continue

                lower_bound = torch.quantile(non_zero_vals, self.lower_p / 100.0)
                upper_bound = torch.quantile(non_zero_vals, self.upper_p / 100.0)
                
                # Clip the image tensor to the calculated bounds
                d[key] = torch.clamp(img_tensor, lower_bound, upper_bound)
        return d
    
import SimpleITK as sitk
import numpy as np
import random
from typing import Dict, Any, Optional
from monai.transforms import MapTransform

import SimpleITK as sitk
import numpy as np
import random
from typing import Dict, Any
from monai.transforms import MapTransform

class LoadSlice(MapTransform):
    """
    An OPTIMIZED transform to load a 2D slice from a 3D NIfTI file.
    It loads the 3D volume into memory ONCE, finds all valid slice indices
    using fast, vectorized NumPy operations, and then randomly selects one.
    """
    def __init__(self, keys: list, axis: int = 2, label_key: str = 'label'):
        """
        Args:
            keys: Keys in the data dictionary to load slices for.
            axis: The axis from which to extract the slice (0=sag, 1=cor, 2=ax).
            label_key: The key corresponding to the label data.
        """
        super().__init__(keys)
        # Mapping from SimpleITK axis (X, Y, Z) to NumPy array axis (Z, Y, X)
        self.sitk_to_np_axis = {0: 2, 1: 1, 2: 0} 
        self.np_axis = self.sitk_to_np_axis[axis]
        self.label_key = label_key
        self.axis = axis  # Store the original axis for reference

    def _get_valid_indices(self, volume_np: np.ndarray) -> np.ndarray:
        """Finds indices of non-blank slices using vectorized numpy."""
        # Sum over the spatial dimensions (H, W) of the slices
        sum_axes = tuple(i for i in range(volume_np.ndim) if i != self.np_axis)
        slice_sums = np.sum(volume_np, axis=sum_axes)
        # Get the indices where the sum is greater than 0
        return np.where(slice_sums > 0)[0]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        
        # --- Step 1: Load all required volumes into memory first ---
        # This minimizes disk I/O to one read per file path.
        sitk_volumes = {
            key: sitk.ReadImage(d[key]) 
            for key in self.keys if key in d and d[key] != 'N/A'
        }
        
        # --- Step 2: Determine the best set of valid slice indices ---
        valid_indices = []
        # Prioritize finding a slice with a label
        if self.label_key in sitk_volumes:
            label_np = sitk.GetArrayFromImage(sitk_volumes[self.label_key])
            valid_indices = self._get_valid_indices(label_np)
        
        # If no labeled slices are found (or no label exists), fallback to image
        if len(valid_indices) == 0 and 'image' in sitk_volumes:
            image_np = sitk.GetArrayFromImage(sitk_volumes['image'])
            valid_indices = self._get_valid_indices(image_np)
        
        # --- Step 3: Choose a slice index ---
        if len(valid_indices) > 0:
            # Randomly choose from the list of good slices
            slice_idx = random.choice(valid_indices)
        else:
            # Last resort: if the entire volume is blank, just pick a random slice
            depth = sitk_volumes['image'].GetSize()[self.axis]
            slice_idx = random.randint(0, depth - 1)
            
        # --- Step 4: Extract the chosen slice from each volume ---
        for key, sitk_vol in sitk_volumes.items():
            volume_np = sitk.GetArrayFromImage(sitk_vol)
            # Use np.take to select the slice along the correct axis
            d[key] = np.take(volume_np, slice_idx, axis=self.np_axis)
                
        return d


class OnTheFly2DDataset(Dataset):
    """
    An efficient 2D Dataset that loads slices on-the-fly.
    Generates two different augmented views for contrastive learning if enabled.
    """
    def __init__(self, hf_dataset, patch_size=(224, 192), is_train=True, is_contrastive=False, has_label=True):
        self.is_train = is_train
        self.patch_size = patch_size
        self.is_contrastive = is_contrastive
        self.has_label = has_label

        self.data_dicts = []
        for item in hf_dataset:
            if item.get('image_path') and item['image_path'] != "N/A":
                self.data_dicts.append({
                    "image": item['image_path'],
                    "label": item.get('label_path', 'N/A'),
                    "label1": item.get('label_path1', 'N/A')
                })

        if not self.data_dicts:
            raise ValueError("hf_dataset did not yield any valid image paths.")

        # --- Initialize two distinct transform pipelines ---
        self.weak_transforms = self._get_weak_transforms()
        if self.is_contrastive:
            self.strong_transforms = self._get_strong_transforms()

    def _get_base_transforms(self):
        """Common transforms for both pipelines (loading and initial formatting)."""

        return [
            LoadSlice(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel", allow_missing_keys=True),
            EnsureTyped(keys="image", dtype=torch.float32),
            EnsureTyped(keys="label", dtype=torch.int8, allow_missing_keys=True),
            Lambdad(
            keys="label", 
            func=lambda x: torch.clamp(x, min=0, max=NUM_CLASSES - 1),
            allow_missing_keys=True
                ),
            ClipIntensityPercentiled(keys="image", lower_percentile=2, upper_percentile=98),
            SafeNormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
        ]

    def _get_weak_transforms(self):
        """Standard augmentations for training (view x) or validation."""
        xforms = self._get_base_transforms()
        
        if self.is_train:
            xforms.extend([
                ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=self.patch_size, allow_missing_keys=True),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1, allow_missing_keys=True), # Horizontal flip
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1), allow_missing_keys=True),
            ])
        else: # For validation, just resize.
            xforms.append(ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=self.patch_size, allow_missing_keys=True))
        xforms.append(SafeNormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
        return Compose(xforms)

    def _get_strong_transforms(self):
        """Strong augmentations for the second contrastive view (x')."""
        
        albumentations_xforms = A.Compose([
            A.ToRGB(p=1.0),
            A.ColorJitter(brightness=0.6, contrast=0.2, p=0.8),
            A.ToGray(p=1.0, num_output_channels=1),
        ])

        def apply_albumentations(img_numpy):
            img_numpy = img_numpy.squeeze(0).numpy()  # Remove channel dimension if present
            augmented = albumentations_xforms(image=img_numpy)['image']
            return torch.from_numpy(augmented).unsqueeze(0)  # Add channel dimension back

        xforms = self._get_base_transforms()
        xforms.extend([
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=self.patch_size, allow_missing_keys=True),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1, allow_missing_keys=True), # Horizontal flip
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1), allow_missing_keys=True),
            RandAffined(
                keys=["image", "label"],
                prob=0.8,
                scale_range=((0.8, 1.2), (0.8, 1.2)), 
                translate_range=(self.patch_size[0] * 0.25, self.patch_size[1] * 0.25),
                rotate_range=(np.pi / 12,), # Rotate up to 30 degrees
                mode=("bilinear", "nearest"),
                padding_mode="reflection",
                allow_missing_keys=True
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=self.patch_size, allow_missing_keys=True),
            Lambdad(keys="image", func=apply_albumentations),
            RandGaussianNoised(keys="image", std=0.1, prob=0.1),
            SafeNormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])
        return Compose(xforms)
    
    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        item_dict = self.data_dicts[idx].copy()
        
        label_path_to_use = None
        if self.is_train and item_dict.get("label1") and item_dict["label1"] != "N/A":
            chosen_key = random.choice(["label", "label1"])
            label_path_to_use = item_dict[chosen_key]
        else:
            label_path_to_use = item_dict.get("label", "N/A")
            
        clean_dict = {"image": item_dict["image"], "label": label_path_to_use}
        if clean_dict["label"] == "N/A":
            del clean_dict["label"]

        if self.is_contrastive:
            processed_data1 = self.strong_transforms(clean_dict)
            processed_data2 = self.strong_transforms(clean_dict)
            if self.has_label and "label" in processed_data1:
                return {
                    "image": processed_data1["image"],
                    "image2": processed_data2["image"],
                    "label": processed_data1["label"],
                }
            else:
                return {
                    "image": processed_data1["image"],
                    "image2": processed_data2["image"],
                    "label": torch.tensor([]),
                }
        else: 
            processed_data1 = self.weak_transforms(clean_dict)
            if self.has_label and "label" in processed_data1:
                return {
                    "image": processed_data1["image"],
                    "label": processed_data1["label"],
                }
            else:
                return {
                    "image": processed_data1["image"],
                    "label": torch.tensor([]), 
                }



class Flare3DPatchDataset(Dataset):
    """
    Robust 3D Dataset for the FLARE challenge using 3D patch sampling.
    """
    def __init__(self, hf_dataset, patch_size=(224, 192, 40), is_train=True, is_contrastive=False):
        # We no longer need the hf_dataset object directly, but a list of file paths
        # This format is required by MONAI's LoadImaged
        self.file_list = [
            {"image": item['image_path'], "label": item['label_path']}
            for item in hf_dataset
        ]
        self.patch_size = patch_size
        self.is_train = is_train
        self.is_contrastive = is_contrastive

        # --- THIS IS THE NEW, UNIFIED 3D TRANSFORMATION PIPELINE ---
        self.transforms = self.get_transforms()
        # -----------------------------------------------------------

    def get_transforms(self):
        """Builds and returns the MONAI transformation pipeline for 3D patches."""
        keys = ["image", "label"]

        # Initial transforms to load and format the data correctly
        initial_transforms = [
            LoadImaged(keys=keys, reader=NibabelReader()),  # Use NibabelReader for 3D NIfTI files
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),  # Ensures data is in (C, D, H, W) format
            EnsureTyped(keys=keys), # Ensures data is in (C, D, H, W) format 
        ]

        # --- If it's the training set, add smart patch sampling and augmentations ---

        if self.is_train:
            train_transforms = [
                        RandCropByPosNegLabeld(
                                            keys=keys,
                                            label_key="label",
                                            spatial_size=self.patch_size,
                                            pos=1,
                                            neg=1,
                                            num_samples=1,
                                            image_key="image",
                                            image_threshold=0,
                                        ),
                        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                        SafeNormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                    ]
            return Compose(initial_transforms + train_transforms)
            # return Compose(initial_transforms)
        
        # --- If it's the validation set, we only load, format, and normalize ---
        # The cropping for validation/inference is often handled differently (e.g., sliding window)
        # For a simple validation check, we'll normalize the whole volume.
        # A more advanced pipeline would crop a central patch here.
        else:
            val_transforms = [
                RandCropByPosNegLabeld(keys=keys,
                                    label_key="label",
                                    spatial_size=self.patch_size,
                                    pos=1,
                                    neg=1,
                                    num_samples=1,
                                    image_key="image",
                                    image_threshold=0),
                SafeNormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
            return Compose(initial_transforms + val_transforms)
            # return Compose(initial_transforms)


    def __len__(self):
        # In a patch-based approach, one epoch can be defined as seeing one patch from each volume
        return len(self.file_list)

    def __getitem__(self, idx):
        # The MONAI pipeline handles everything from loading to final tensor creation
        
        data_dict = self.transforms(self.file_list[idx])
        # --- IMPORTANT CHANGE: NO MORE ONE-HOT ENCODING ---
        # 3D models (like 3D U-Net) and MONAI's loss functions (like DiceCELoss)
        # expect the label mask to be a single-channel tensor with integer class labels.
        # Shape: (1, D, H, W), not (C, D, H, W). This simplifies the code greatly.
        # if self.is_train:
        image = data_dict[0]['image'].permute(0, 3, 1, 2)  # (C, D, H, W) -> (D, H, W, C)
        label = data_dict[0]['label'].long().permute(0, 3, 1, 2)  # (C, D, H, W) -> (D, H, W, C)
        # else:
        #     print("----------- Validation Data -----------")
        #     print(data_dict[0]['image'].shape, data_dict[0]['label'].shape)
        #     image = data_dict[0]['image'].permute(0, 3, 1, 2)  # (C, D, H, W) -> (D, H, W, C)
        #     label = data_dict[0]['label'].long().permute(0, 3, 1, 2)  # (C, D, H, W) -> (D, H, W, C)
        
        if self.is_contrastive:
             # For contrastive, we need to apply augmentations again on the same initial data
             # Note: This is less efficient, better to have a custom transform.
             # But for clarity, we re-run the pipeline.
            data_dict2 = self.transforms(self.file_list[idx])
            image2 = data_dict2['image']
            label2 = data_dict2['label'].long()
            return image, image2, label, label2
        
        # For validation, we return the whole volume. You would use a sliding window inference
        # function (like monai.inferers.sliding_window_inference) on this output.
        if not self.is_train:
             return {"image": image, "label": label, "path": self.file_list[idx]['image']}

        return {"image": image, "label": label}
    
    
# Example usage:
if __name__ == "__main__":
#     # Assuming hf_dataset is already defined and loaded
    patch_size = (224, 192)  # Example patch size
    hf_dataset = load_dataset("./local_flare_loader.py", name="validation_mri", data_dir="/scratch/work/zhul2/data/FLARE-MedFM/FLARE-Task3-DomainAdaption", trust_remote_code=True)["train"]
    dataset = OnTheFly2DDataset(hf_dataset, patch_size=patch_size, is_train=False, is_contrastive=False, has_label=True)
    print(f"Dataset length: {len(dataset)}")
    for i in range(len(dataset)):
        samples = dataset[i]
        print(samples["image"].shape, samples["label"].shape)
        break
        # if len(torch.unique(sample['label'])) > 14:
        #     print(f"Sample {i} has more than 14 classes in label: {torch.unique(sample['label'])}")
        # Print the shapes of the image and label tensors
        # if 'label' in sample:
        #     print(f"Sample {i} - Image shape: {sample['image'].shape}, Label shape: {sample['label'].shape}, Label unique values: {torch.unique(sample['label']).size()}")
        # else:
        #     print(f"Sample {i} - Image shape: {sample['image'].shape}, No label available")
        # Uncomment to see the actual data
        # print(sample)

