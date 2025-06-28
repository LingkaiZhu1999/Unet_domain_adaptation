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
    RandHistogramShiftd,
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
    def __init__(self, keys: list, axis: int = 2, label_key: str = 'label', num_classes: int = NUM_CLASSES, slice_based: bool = False):
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
        self.slice_based = slice_based
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
        if self.slice_based:
            # sample two slices
            slice_idx1 = random.choice(valid_indices)
            # sample based on gaussian distribution
            slice_idx2 = int(np.random.normal(loc=slice_idx1, scale=0.5))
            # Ensure slice_idx2 is within bounds
            slice_idx2 = max(0, min(slice_idx2, len(valid_indices) - 1))
            # Store both indices in the dictionary
            for key, volume_np in np_volumes.items():
                slice1 = np.take(volume_np, slice_idx1, axis=self.np_axis)
                slice2 = np.take(volume_np, slice_idx2, axis=self.np_axis)
                d[key] = slice1
                if key == "image":
                    d["image2"] = slice2
        else:
            slice_idx = random.choice(valid_indices)
            for key, volume_np in np_volumes.items():
                slice_data = np.take(volume_np, slice_idx, axis=self.np_axis)
                d[key] = slice_data
        return d


class OnTheFly2DDataset(Dataset):
    """
    An efficient 2D Dataset that loads slices on-the-fly.
    Generates two different augmented views for contrastive learning if enabled.
    """
    def __init__(self, hf_dataset, patch_size=(224, 192), is_train=True, is_contrastive=False, has_label=True, slice_based=False):
        self.is_train = is_train
        self.patch_size = patch_size
        self.is_contrastive = is_contrastive
        self.has_label = has_label
        self.slice_based = slice_based

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
        # Patch-wise Gaussian noise
        if self.slice_based:
            xforms = [
                LoadSlice(keys=["image", "image2", "label"], slice_based=True),
            ]
        else:
            xforms = [
                LoadSlice(keys=["image", "label"], slice_based=False),
            ]
        xforms.extend([EnsureChannelFirstd(keys=["image", "image2", "label"], channel_dim="no_channel", allow_missing_keys=True),
            EnsureTyped(keys=["image", "image2"], dtype=torch.float32, allow_missing_keys=True),
            EnsureTyped(keys="label", dtype=torch.int8, allow_missing_keys=True),
            Lambdad(
            keys="label", 
            func=lambda x: torch.clamp(x, min=0, max=NUM_CLASSES - 1),
            allow_missing_keys=True
                ),
            ClipIntensityPercentiled(keys=["image", "image2"], lower_percentile=5, upper_percentile=95),
            Resized(keys=["image", "image2", "label"], spatial_size=(512, 512), allow_missing_keys=True)])
        return Compose(xforms)

    def _get_weak_transforms(self):
        """Standard augmentations for training (view x) or validation."""
        xforms = []

        if self.is_train:
            xforms.extend([
                RandSpatialCropd(keys=["image", "label", "image2"], roi_size=self.patch_size, allow_missing_keys=True),
                RandFlipd(keys=["image", "image2", "label"], prob=0.5, spatial_axis=1, allow_missing_keys=True), # Horizontal flip
                RandRotate90d(keys=["image", "label", "image2"], prob=0.5, max_k=3, spatial_axes=(0, 1), allow_missing_keys=True),
            ])
        xforms.append(NormalizeIntensityd(keys=["image", "image2"], nonzero=True, channel_wise=True, allow_missing_keys=True))
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
            RandGaussianSmoothd(keys="image", sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), prob=prob_intensity_appearance),
            RandScaleIntensityd(keys="image", factors=0.1, prob=prob_intensity_appearance),
            RandAdjustContrastd(keys="image", gamma=(0.5, 2.0), prob=prob_intensity_appearance),
            # RandShiftIntensityd(keys="image", offsets=(-0.1, 0.1), prob=prob_intensity_appearance),
            # RandHistogramShiftd(keys="image", num_control_points=(5, 15), prob=prob_intensity_appearance)
            ]),
            
             # --- Noise and Dropout ---
            RandGaussianNoised(keys="image", std=0.01, prob=prob_noise),
            RandCoarseDropoutd(
                keys=["image", "label"],
                holes=1, max_holes=3,
                spatial_size=(16, 16), max_spatial_size=(32, 32),
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
            # process (image and label), and image2 seperately for strong transforms
            if "image2" in processed_data:
                if self.has_label and "label" in processed_data:
                    processed_data1 = self.strong_transforms({"image": processed_data["image"], "label": processed_data["label"]})
                else:
                    processed_data1 = self.strong_transforms({"image": processed_data["image"]})
                processed_data2 = self.strong_transforms({"image": processed_data["image2"]})
            else:
                processed_data1 = self.strong_transforms({"image": processed_data["image"], "label": processed_data["label"]})
                processed_data2 = self.strong_transforms({"image": processed_data["image"]})
            # Apply the strong transforms to generate two augmented views
            
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

    
    
# Example usage:
if __name__ == "__main__":
#     # Assuming hf_dataset is already defined and loaded
    patch_size = (224, 224)  # Example patch size
    hf_dataset = load_dataset("./local_flare_loader.py", name="train_ct_gt", data_dir="/scratch/work/zhul2/data/FLARE-MedFM/FLARE-Task3-DomainAdaption", trust_remote_code=True)["train"]
    # dataset = OnTheFly2DDataset(hf_dataset, patch_size=patch_size, is_train=True, is_contrastive=True, has_label=True)
    dataset = OnTheFly2DDataset(
        hf_dataset,
        patch_size=patch_size, is_train=True, is_contrastive=True, has_label=True, slice_based=True
    )
    # plot image, image2
    # visualize a batch of images
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    images = batch["image"]
    if "image2" not in batch:
        batch["image2"] = images
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
        axs[2].imshow(labels[0].squeeze().cpu().numpy())
    plt.show()
    plt.savefig("example_batch_mri.png")

