import torch
# from torch.utils.data import Dataset
from monai.data import Dataset
import SimpleITK as sitk
import warnings
import numpy as np
import random
from typing import Dict, Any, Optional
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
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandomOrder,
    RandCoarseDropoutd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    Resized,
    RandZoomd,
)

from monai.data import NibabelReader
import albumentations as A
from monai.transforms import Lambdad
import torch
import random
from typing import Dict, Union, List, Any
from datasets import load_dataset
import numpy as np
import os
import json
import atexit
import threading
import torch.distributions as dist
# --- STEP 1: Create a simple, self-contained custom transform ---

NUM_CLASSES = 14  # Number of classes in the FLARE-3D dataset

# In dataset.py, near the top
from monai.transforms import MapTransform, Resized

class ConditionalResizeSmaller(MapTransform):
    """
    A MapTransform that applies Resized to the specified keys only if the
    image's spatial size is smaller than the target spatial_size in any dimension.
    """
    def __init__(self, keys: list, spatial_size: tuple, allow_missing_keys: bool = False):
        """
        Args:
            keys: Keys to apply the resizing to.
            spatial_size: The target spatial size for resizing.
            allow_missing_keys: Corresponds to the parameter in super class.
        """
        super().__init__(keys, allow_missing_keys)
        self.spatial_size = spatial_size
        # Create a single instance of the Resized transform to reuse
        self.resizer = Resized(keys=keys, spatial_size=spatial_size, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        # We need to make a decision based on the image's shape.
        # Assuming the first key is the image or a reference with the same shape.
        ref_key = self.keys[0] 
        
        if ref_key not in d:
            return d # Do nothing if the reference key is missing

        img_shape = d[ref_key].shape[1:] # Get (H, W) from (C, H, W)
        
        # Check if any dimension is smaller than the target size
        is_smaller = any(img_dim < target_dim for img_dim, target_dim in zip(img_shape, self.spatial_size))

        if is_smaller:
            # If it's smaller, apply the pre-configured resizer transform to the whole dictionary
            return self.resizer(d)
        else:
            # Otherwise, return the data dictionary unchanged
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
        assert 0 <= self.lower_p < self.upper_p <= 100, "Percentiles must be in the range [0, 100] and lower < upper."

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



class LoadSlice(MapTransform):
    """
    An OPTIMIZED transform to load a 2D slice from a 3D NIfTI file.
    It loads the 3D volume into memory ONCE, finds all valid slice indices
    using fast, vectorized NumPy operations, and then randomly selects one.
    """
    def __init__(self, keys: list, axis: int = 2, label_key: str = 'label', num_classes: int = NUM_CLASSES):
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
        self.num_classes = num_classes
        # normalize the categorical logits
        self.categorical_logits = 70. / torch.tensor([1201., 135., 189., 74., 70., 43., 2., 3., 26., 8., 176., 41., 139.])
        self.categorical_logits = torch.concat((torch.tensor([0.01]), self.categorical_logits))  # Add a small weight for the background class
        self.categorical_logits = torch.clamp(self.categorical_logits, max=10.0, min=0.1)  # Clamp weights to avoid too high or too low values
        self.categorical_logits = self.categorical_logits / self.categorical_logits.sum()

    def _compute_valid_indices(self, volume_np: np.ndarray, is_label: bool) -> np.ndarray:
        """Computes valid indices using the std method. (Your existing logic)"""
        if volume_np.ndim != 3: return np.array([])
        sum_axes = tuple(i for i in range(volume_np.ndim) if i != self.np_axis)
        # get number of unique values in the slice
        if is_label and self.num_classes is not None:
            # Generate the list of expected class values (e.g., [0, 1, ..., 13])
            # all_class_values = [6, 8]
            # print(all_class_values)
            # define a probability distribution for the classes
            categorical = dist.Categorical(probs=self.categorical_logits)
            include_class_values = categorical.sample((1,)).tolist()

            # For each class, find which slices contain it.
            # (volume_np == c) creates a 3D boolean mask for class c.
            # .any(axis=sum_axes) collapses the slice dimensions, resulting in a 1D
            # boolean array indicating if class c is present in each slice.
            slice_has_class = [
                (volume_np == c).any(axis=sum_axes) for c in include_class_values
            ]

            # Stack the 1D arrays into a 2D array (num_classes, num_slices)
            stacked_presences = np.stack(slice_has_class, axis=0)

            # A slice is valid only if it has ALL classes.
            # np.all(axis=0) finds slices where every class is present.
            slice_has_all_classes = np.all(stacked_presences, axis=0)
            # Get the indices where the condition is True
            valid_indices = np.where(slice_has_all_classes)[0]
            return valid_indices
        else:
            slice_stds = np.std(volume_np, axis=sum_axes)
        return np.where(slice_stds > 0.001)[0]
    

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        image_path = os.path.abspath(d["image"])
        sitk_volumes = {key: sitk.ReadImage(d[key]) for key in self.keys if key in d and d[key] != 'N/A'}
        np_volumes = {key: sitk.GetArrayFromImage(vol) for key, vol in sitk_volumes.items()}
        valid_indices = []
        if self.label_key in np_volumes:
            valid_indices = self._compute_valid_indices(np_volumes[self.label_key], is_label=True)

        if len(valid_indices) == 0:
            valid_indices = self._compute_valid_indices(np_volumes['image'], is_label=False)
        if len(valid_indices) == 0:
            raise ValueError(f"No valid slices found in {image_path} for label key {self.label_key}.")
        
        # if stik_volumes and np_volumes exists
        # --- Step 4: Choose a slice index ---
        if len(valid_indices) > 0:
            slice_idx = random.choice(valid_indices)
        else:
            # Fallback: if entire volume is blank, get depth from original sitk object
            # Using 'image' as the reference for size.
            raise ValueError(f"No valid slices found in {image_path}. The volume may be empty or all slices have zero variance.")
            
        # --- Step 5: Extract the chosen slice from each NumPy volume ---
        for key, volume_np in np_volumes.items():
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

        self.base_transforms = self._get_base_transforms()
        # --- Initialize two distinct transform pipelines ---
        self.weak_transforms = self._get_weak_transforms()
        if self.is_contrastive:
            self.strong_transforms = self._get_strong_transforms()

    def _get_base_transforms(self):
        """Common transforms for both pipelines (loading and initial formatting)."""

        return Compose([
            LoadSlice(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel", allow_missing_keys=True),
            EnsureTyped(keys="image", dtype=torch.float32),
            EnsureTyped(keys="label", dtype=torch.int8, allow_missing_keys=True),
            Lambdad(
            keys="label", 
            func=lambda x: torch.clamp(x, min=0, max=NUM_CLASSES - 1),
            allow_missing_keys=True
                ),
            ClipIntensityPercentiled(keys="image", lower_percentile=0.5, upper_percentile=99.5),
            Resized(keys=["image", "label"], spatial_size=(512, 512), allow_missing_keys=True),
        ])

    def _get_weak_transforms(self):
        """Standard augmentations for training (view x) or validation."""
        xforms = []

        if self.is_train:
            xforms.extend([
                RandSpatialCropd(keys=["image", "label"], roi_size=self.patch_size, allow_missing_keys=True),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1, allow_missing_keys=True), # Horizontal flip
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1), allow_missing_keys=True),
            ])
        xforms.append(NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
        return Compose(xforms)

    def _get_strong_transforms(self):
        """Strong augmentations for the second contrastive view (x')."""
        
        xforms = []
        prob_intensity_appearance = 0.8
        prob_shape = 1.0
        prob_noise = 0.2
        prob_drop = 0.2

        # xforms.extend([RandSpatialCropd(keys=["image", "label"], roi_size=self.patch_size, random_size=False, allow_missing_keys=True)])

        
        xforms.extend([
            # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=self.patch_size, allow_missing_keys=True),
            # CropForegroundd(keys=["image", "label"], source_key="label", allow_missing_keys=True),

            RandFlipd(keys=["image", "label"], prob=prob_shape, spatial_axis=1, allow_missing_keys=True), # Horizontal flip
            RandRotate90d(keys=["image", "label"], prob=prob_shape, max_k=3, spatial_axes=(0, 1), allow_missing_keys=True),
            RandAffined(
                keys=["image", "label"],
                prob=prob_shape,
                # scale_range=((0, 0.15), (0, 0.15)), 
                translate_range=(self.patch_size[0] * 0.15, self.patch_size[1] * 0.15),
                # rotate_range=(np.pi / 12,), # Rotate up to 30 degrees
                mode=("bilinear", "nearest"),
                padding_mode="reflection",
                allow_missing_keys=True
            ),
            RandZoomd(
                keys=["image", "label"],
                prob=prob_shape,
                min_zoom=1.0,
                max_zoom=1.5,
                mode=("bilinear", "nearest"),
                padding_mode="reflection",
                allow_missing_keys=True,
            ),
            # --- Intensity and Appearance Augmentations (applied in a random order) ---
            RandomOrder([
            RandGaussianSmoothd(keys="image", sigma_x=(1, 2), sigma_y=(1, 2), prob=prob_intensity_appearance),
            RandScaleIntensityd(keys="image", factors=0.1, prob=prob_intensity_appearance),
            RandAdjustContrastd(keys="image", gamma=(0.75, 1.25), prob=prob_intensity_appearance),
            ]),
            
             # --- Noise and Dropout ---
            RandGaussianNoised(keys="image", std=0.01, prob=prob_noise),
            RandCoarseDropoutd(
                keys=["image", "label"],
                holes=1, max_holes=3,
                spatial_size=(8, 8), max_spatial_size=(16, 16),
                fill_value=0, # Use 0 for background
                prob=prob_drop,
                allow_missing_keys=True
            ), 

            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
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
            processed_data = self.base_transforms(clean_dict)
            # Apply the strong transforms to generate two augmented views
            processed_data1 = self.strong_transforms(processed_data)
            processed_data2 = self.strong_transforms(processed_data)
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
            processed_data1 = self.weak_transforms(self.base_transforms(clean_dict))
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



class Flare2DDataset(Dataset):
    """
    A 2D Dataset for the FLARE challenge using preprocessed 2D slices saved as .npy files.
    This dataset is designed for training and validation, with optional contrastive learning support.
    """
    def __init__(self, data_path, patch_size=(224, 192), is_train=True, is_contrastive=False, has_label=True):
        """
        Args:
            data_path (str): Path to the directory containing the preprocessed .npy files, with subdirectories ./images and ./masks.
            with same name as the image files.
            patch_size (tuple): Size of the patches to be extracted from the images.
            is_train (bool): Whether this dataset is for training or validation.
            is_contrastive (bool): Whether to generate two augmented views for contrastive learning.
            has_label (bool): Whether the dataset contains labels.
        """
        self.is_train = is_train
        self.patch_size = patch_size
        self.is_contrastive = is_contrastive
        self.has_label = has_label
        self.data_dicts = []
        # Load all image paths from the specified directory
        image_dir = os.path.join(data_path, "images")
        mask_dir = os.path.join(data_path, "masks") if has_label else None
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory {image_dir} does not exist.")
        image_files = sorted(os.listdir(image_dir))
        for image_file in image_files:
            if image_file.endswith(".npy"):
                image_path = os.path.join(image_dir, image_file)
                label_path = None
                if has_label:
                    label_path = os.path.join(mask_dir, image_file) if mask_dir else None
                self.data_dicts.append({
                    "image": image_path,
                    "label": label_path
                })
        
        self.base_transforms = self._get_base_transforms()
        
    def _get_base_transforms(self):
        """Common transforms for both pipelines (loading and initial formatting)."""
        return Compose([
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel", allow_missing_keys=True),
            EnsureTyped(keys="image", dtype=torch.float32),
            EnsureTyped(keys="label", dtype=torch.int8, allow_missing_keys=True),
            Lambdad(
                keys="label", 
                func=lambda x: torch.clamp(x, min=0, max=NUM_CLASSES - 1),
                allow_missing_keys=True
            ),
            ClipIntensityPercentiled(keys="image", lower_percentile=0.5, upper_percentile=99.5),
            Resized(keys=["image", "label"], spatial_size=(512, 512), allow_missing_keys=True),
        ])
    
    def _get_weak_transforms(self):
        """Standard augmentations for training (view x) or validation."""
        xforms = []

        if self.is_train:
            xforms.extend([
                RandSpatialCropd(keys=["image", "label"], spatial_size=self.patch_size, allow_missing_keys=True),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1, allow_missing_keys=True), # Horizontal flip
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1), allow_missing_keys=True),
            ])
        xforms.append(NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
        return Compose(xforms)
    
    def __len__(self):
        return len(self.data_dicts)
    
    def __getitem__(self, idx):
        # first read the image and label from the file paths
        item_dict = self.data_dicts[idx].copy()
        image_path = item_dict["image"]
        label_path = item_dict.get("label", "N/A")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        # Load the image and label as numpy arrays
        image_np = np.load(image_path)
        if label_path != "N/A" and os.path.exists(label_path):
            label_np = np.load(label_path)
        else:
            label_np = np.array([])  # Empty array if no label is provided
        # Create a dictionary to hold the data
        data_dict = {"image": image_np, "label": label_np}
        # Apply the base transforms
        processed_data = self.base_transforms(data_dict)
        # Apply the weak transforms for training or validation
        processed_data_weak = self._get_weak_transforms()(processed_data)
        return {
            "image": processed_data_weak["image"],
            "label": processed_data_weak.get("label", torch.tensor([]))}
        
        
            
            
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
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
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
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
            return Compose(initial_transforms + val_transforms)


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
    patch_size = (224, 224)  # Example patch size
    hf_dataset = load_dataset("./local_flare_loader.py", name="train_ct_gt", data_dir="/scratch/work/zhul2/data/FLARE-MedFM/FLARE-Task3-DomainAdaption", trust_remote_code=True)["train"]
    # dataset = OnTheFly2DDataset(hf_dataset, patch_size=patch_size, is_train=True, is_contrastive=True, has_label=True)
    dataset = OnTheFly2DDataset(
        hf_dataset,
        patch_size=patch_size, is_train=True, is_contrastive=True, has_label=True
    )
    # plot image, image2
    # visualize a batch of images
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    images = batch["image"]
    images2 = batch["image2"]
    labels = batch["label"] if "label" in batch else torch.tensor([])
    print("unique labels:", len(torch.unique(labels)))
    print(f"Batch size: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Image2 shape: {images2[0].shape}")
    print(f"Label shape: {labels[0].shape}")
    # Plot the first image and its corresponding label
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(images[0].squeeze().cpu().numpy(), cmap='gray')
    axs[0].set_title("Image 1")
    axs[1].imshow(images2[0].squeeze().cpu().numpy(), cmap='gray')
    axs[1].set_title("Image 2")
    if labels.numel() > 0:
        axs[2].imshow(labels[0].squeeze().cpu().numpy(), cmap='gray')
    plt.show()
    plt.savefig("example_batch.png")

