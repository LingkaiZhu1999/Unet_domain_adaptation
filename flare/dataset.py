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
    RandGaussianSharpend, 
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    RandGridDistortiond,
    RandRicianNoised,
    Identityd,
    ClipIntensityPercentilesd,
)

from monai.data import NibabelReader
import albumentations as A
from monai.transforms import Lambdad
import torch
import random
from typing import Dict, Union, List, Any, Tuple
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
from monai.config import DtypeLike
from monai.utils import convert_to_numpy

class N4BiasCorrectiond(MapTransform):
    """
    A MONAI-style transform to wrap SimpleITK's N4BiasFieldCorrectionImageFilter.
    Operates on 3D image tensors in a dictionary.
    
    Args:
        keys: The key(s) of the image to transform.
        mask_key: The key for the foreground mask. N4 works best when it knows
                  where the foreground is. If not provided, it will use
                  Otsu thresholding to create one.
    """
    def __init__(self, keys: str, mask_key: str = None):
        super().__init__(keys)
        self.mask_key = mask_key
        # Initialize the filter here to reuse it
        self.n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
        self.n4_filter.SetMaximumNumberOfIterations([50] * 4) # A common setting

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            img_tensor = d[key]
            
            # --- 1. Convert Tensor to SimpleITK Image ---
            # MONAI images are channel-first (C, H, W, D), SITK is (W, H, D)
            # We assume a single channel for bias field correction.
            img_np = convert_to_numpy(img_tensor[0, ...], wrap_sequence=True)
            sitk_image = sitk.GetImageFromArray(img_np)
            
            # Preserve metadata (spacing, origin)
            if f"{key}_meta_dict" in d:
                meta = d[f"{key}_meta_dict"]
                if 'spacing' in meta:
                    sitk_image.SetSpacing(meta['spacing'])

            # --- 2. Create the Mask ---
            if self.mask_key and self.mask_key in d:
                # Use the provided label/mask
                mask_np = convert_to_numpy(d[self.mask_key][0, ...], wrap_sequence=True).astype(np.uint8)
                sitk_mask = sitk.GetImageFromArray(mask_np)
                sitk_mask.CopyInformation(sitk_image) # Ensure mask and image have same space
            else:
                # Or create one automatically using Otsu
                sitk_mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)

            # --- 3. Run the N4 Filter ---
            corrected_sitk_image = self.n4_filter.Execute(sitk_image, sitk_mask)

            # --- 4. Convert back to NumPy array and update dictionary ---
            corrected_np = sitk.GetArrayFromImage(corrected_sitk_image)
            
            # Add the channel dimension back and update the dictionary entry
            d[key] = np.expand_dims(corrected_np, axis=0).astype(img_tensor.dtype)

        return d
    
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
    

class LoadSlice(MapTransform):
    """
    An OPTIMIZED transform to load a 2D slice from a 3D NIfTI file.
    It loads the 3D volume into memory ONCE, finds all valid slice indices
    using fast, vectorized NumPy operations, and then randomly selects one.
    """
    def __init__(self, keys: list, axis: int = 2, label_key: str = 'label', num_classes: int = NUM_CLASSES, slice_based: bool = False, 
                 target_spacing: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0)):
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
        self.target_spacing = target_spacing
        self.image_interpolator = sitk.sitkLinear
        self.label_interpolator = sitk.sitkNearestNeighbor 

    def _resample_volume(self, sitk_volume: sitk.Image, interpolator: int) -> sitk.Image:
        """
        Resamples a SimpleITK Image to the target spacing.
        """
        original_spacing = sitk_volume.GetSpacing()
        original_size = sitk_volume.GetSize()
        
        # Calculate the new size based on the ratio of original and target spacing
        new_size = [
            int(round(orig_sz * orig_spc / targ_spc))
            for orig_sz, orig_spc, targ_spc in zip(original_size, original_spacing, self.target_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(sitk_volume.GetDirection())
        resampler.SetOutputOrigin(sitk_volume.GetOrigin())
        resampler.SetTransform(sitk.Transform()) # Use identity transform
        resampler.SetDefaultPixelValue(sitk_volume.GetPixelIDValue()) # Use input's min value as default
        resampler.SetInterpolator(interpolator)

        return resampler.Execute(sitk_volume)
    
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
        
        if self.target_spacing is not None:
            resampled_volumes = {}
            for key, vol in sitk_volumes.items():
                # Choose the correct interpolator (Nearest for labels, Linear for images)
                interpolator = self.label_interpolator if key == self.label_key else self.image_interpolator
                resampled_volumes[key] = self._resample_volume(vol, interpolator)
            # The rest of the pipeline will use the resampled volumes
            sitk_volumes_to_process = resampled_volumes
        else:
            # If no resampling, just use the original volumes
            sitk_volumes_to_process = sitk_volumes
            
        np_volumes = {key: sitk.GetArrayFromImage(vol) for key, vol in sitk_volumes_to_process.items()}
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
            slice_idx2 = int(np.random.normal(loc=slice_idx1, scale=1))
            # Ensure slice_idx2 is within bounds
            slice_idx2 = max(0, min(slice_idx2, np_volumes['image'].shape[self.np_axis] - 1))
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
    def __init__(self, hf_dataset, patch_size=(224, 192), is_train=True, is_contrastive=False, has_label=True, slice_based=False, volume_type="ct"):
        self.is_train = is_train
        self.patch_size = patch_size
        self.is_contrastive = is_contrastive
        self.has_label = has_label
        self.slice_based = slice_based
        self.volume_type = volume_type
        self.target_spacing = (1.0, 1.0, 1.0) 
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
                LoadSlice(keys=["image", "image2", "label"], slice_based=True, target_spacing=self.target_spacing),
            ]
        else:
            xforms = [
                LoadSlice(keys=["image", "label"], slice_based=False, target_spacing=self.target_spacing),
            ]
        xforms.extend([EnsureChannelFirstd(keys=["image", "image2", "label"], channel_dim="no_channel", allow_missing_keys=True),
            EnsureTyped(keys=["image", "image2"], dtype=torch.float32, allow_missing_keys=True),
            EnsureTyped(keys="label", dtype=torch.int8, allow_missing_keys=True),
            Lambdad(
            keys="label", 
            func=lambda x: torch.clamp(x, min=0, max=NUM_CLASSES - 1),
            allow_missing_keys=True
                ),
            # ClipIntensityPercentiled(keys=["image", "image2"], lower_percentile=5, upper_percentile=95),
            Resized(keys=["image", "image2", "label"], spatial_size=(512, 512), mode=('bilinear', 'bilinear', 'nearest'),
                    allow_missing_keys=True)])
        if self.volume_type == "ct":
            xforms.append(ScaleIntensityRanged(keys=["image", "image2"], a_min=-1000, a_max=1000, allow_missing_keys=True))
            xforms.append(NormalizeIntensityd(keys=["image", "image2"], nonzero=False, channel_wise=True, allow_missing_keys=True))
        elif self.volume_type == "mri":
            xforms.extend([ClipIntensityPercentilesd(keys=["image", "image2"], lower=0.5, upper=99.5, channel_wise=True, allow_missing_keys=True),
                NormalizeIntensityd(keys=["image", "image2"], nonzero=False, channel_wise=True, allow_missing_keys=True)])
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
        return Compose(xforms)

    def _get_strong_transforms(self):
        """Strong augmentations for the second contrastive view (x')."""
        
        xforms = []
        prob_intensity_appearance = 0.5
        prob_shape = 0.5
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
                # scale_range=((0, 0.5), (0, 0.5)), 
                translate_range=(self.patch_size[0] * 0.15, self.patch_size[1] * 0.15),
                rotate_range=(np.pi / 12,), # Rotate up to 30 degrees
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
            # RandGaussianSmoothd(keys="image", sigma_x=(0.5, 2), sigma_y=(0.5, 2), prob=prob_intensity_appearance),
            RandScaleIntensityd(keys="image", factors=0.5, prob=prob_intensity_appearance),
            RandAdjustContrastd(keys="image", gamma=(0.5, 2), prob=prob_intensity_appearance),
            RandShiftIntensityd(keys="image", offsets=(-0.1, 0.1), prob=prob_intensity_appearance),
            RandRicianNoised(keys=["image"], prob=prob_noise, allow_missing_keys=True) if self.volume_type == "ct" else RandGaussianNoised(keys="image", std=0.01, prob=prob_noise)
            # RandHistogramShiftd(keys="image", num_control_points=5, prob=1), 
            # RandGaussianSharpend(keys="image", prob=prob_intensity_appearance)
            ]),
            
             # --- Noise and Dropout ---
            
            # RandCoarseDropoutd(
            #     keys=["image", "label"],
            #     holes=1, max_holes=5,
            #     spatial_size=(16, 16), max_spatial_size=(32, 32),
            #     fill_value=0, # Use 0 for background
            #     prob=prob_drop,
            #     allow_missing_keys=True
            # ), 

            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])

        # xforms.append(NormalizeIntensityd(keys=["image", "image2"], channel_wise=True, allow_missing_keys=True))
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
            if self.is_train:
                processed_data1 = self.weak_transforms(self.base_transforms(clean_dict))
            else:
                processed_data1 = self.base_transforms(clean_dict)
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

# You can keep your NUM_CLASSES constant defined elsewhere
# from your_config import NUM_CLASSES

class Flare3DPatchDataset(Dataset):
    """
    A robust, feature-rich 3D Dataset for the FLARE challenge, inspired by
    advanced 2D data loading techniques. It performs on-the-fly loading of
    3D volumes, applies extensive 3D augmentations, and supports
    contrastive learning by generating two augmented patch views.

    Features:
    - On-the-fly loading of NIfTI files.
    - Voxel spacing normalization to ensure consistent resolution.
    - Foreground cropping to focus on relevant anatomy.
    - Smart patch sampling (Pos/Neg ratio) to avoid empty patches.
    - Separate pipelines for weak and strong augmentations.
    - Full support for 3D contrastive learning.
    - Modality-specific intensity scaling (CT vs. MRI).
    """
    def __init__(
        self,
        hf_dataset,
        patch_size: tuple = (128, 128, 64),
        is_train: bool = True,
        is_contrastive: bool = False,
        num_samples_per_volume: int = 2,
        volume_type: str = "ct",
        target_spacing: tuple = (1.0, 1.0, 1.0),
    ):
        """
        Args:
            hf_dataset: A dataset object (like from Hugging Face) that yields dicts
                        with 'image_path' and 'label_path'.
            patch_size: The size of the 3D patches to extract (H, W, D).
            is_train: If True, applies training augmentations. Otherwise, prepares
                      data for validation (outputs full volume).
            is_contrastive: If True, generates two strongly augmented views ('image' and 'image2').
            num_samples_per_volume: Number of patches to sample from each 3D volume per epoch.
                                    Only active during training.
            volume_type: 'ct' or 'mri'. Determines the intensity scaling method.
            target_spacing: The target voxel spacing (x, y, z) for resampling.
        """
        self.file_list = [
            {"image": item['image_path'], "label": item['label_path']}
            for item in hf_dataset if item.get('image_path')
        ]
        self.patch_size = patch_size
        self.is_train = is_train
        self.is_contrastive = is_contrastive
        self.num_samples_per_volume = num_samples_per_volume if is_train else 1
        self.volume_type = volume_type
        self.target_spacing = target_spacing

        # --- Define transform pipelines based on the 2D dataset's structure ---
        self.base_transforms = self._get_base_transforms()
        self.weak_transforms = self._get_weak_transforms()
        if self.is_contrastive:
            self.strong_transforms = self._get_strong_transforms()

    def _get_base_transforms(self):
        """
        Core pre-processing transforms applied to every volume.
        Loads, formats, resamples, and crops the data.
        """
        keys = ["image", "label"]
        xforms = [
            LoadImaged(keys=keys, reader=NibabelReader(), allow_missing_keys=True),
            EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
            EnsureTyped(keys=keys, dtype=torch.float32, allow_missing_keys=True),
            # --- FEATURE: Voxel Spacing Normalization ---
            Spacingd(
                keys=keys,
                pixdim=self.target_spacing,
                mode=("bilinear", "nearest"), # Linear for image, NN for label
                allow_missing_keys=True,  # Handle cases where label might be missing
            ),
            # --- FEATURE: Modality-Specific Intensity Scaling ---
            # --- FEATURE: Crop to Foreground ---
            # This significantly improves the efficiency of patch sampling.
            CropForegroundd(keys=keys, source_key="image", margin=10, allow_missing_keys=True),
            # --- FEATURE: Resize to Patch Size ---
            
            # RandSpatialCropd(
            #     keys=keys,
            #     roi_size=self.patch_size,  # Resize to a fixed size
            #     allow_missing_keys=True,  # Handle cases where label might be missing
            # ),
            ResizeWithPadOrCropd(keys=keys, spatial_size=self.patch_size, allow_missing_keys=True)
        ]
        if not self.is_train:
            if self.volume_type == "ct":
            # Values typical for abdominal CT scans
                xforms+= [ScaleIntensityRanged(keys="image", a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True)]
            elif self.volume_type == "mri":
                # z-score normalization
                xforms+= [NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)]
        return Compose(xforms)

    def _get_intensity_scaler(self):
        """Returns the appropriate intensity scaling transform based on volume type."""
        if self.volume_type == "ct":
            # Values typical for abdominal CT scans
            return [ScaleIntensityRanged(keys="image", a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True)]
        elif self.volume_type == "mri":
            # z-score normalization
            return [NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)]

    def _get_weak_transforms(self):
        """Standard augmentations for supervised training or validation."""
        keys = ["image", "label"]
        xforms = []
        if self.is_train:
            prob = 1.0
            prob_intensity_appearance = 0.5
            xforms += [
                # --- FEATURE: Smart Patch Sampling ---
                # RandCropByPosNegLabeld(
                #     keys=keys,
                #     label_key="label",
                #     spatial_size=self.patch_size,
                #     pos=0.8, neg=0.2, # Favor patches with labels
                #     num_samples=self.num_samples_per_volume,
                #     image_key="image",
                #     image_threshold=0,
                # ),
                RandAffined(
                    keys=keys,
                    prob=prob,
                    rotate_range=(np.pi/12, np.pi/12, np.pi/12),
                    translate_range=(self.path_size[0] * 0.15, self.path_size[1] * 0.15, self.path_size[2]),
                    mode=("bilinear", "nearest"),
                    allow_missing_keys=True
                ),
                RandGridDistortiond(keys=keys, prob=0.2, num_cells=5, distort_limit=(-0.2, 0.2), allow_missing_keys=True),
                # RandFlipd(keys=keys, prob=prob, spatial_axis=0, allow_missing_keys=True), # Flip along sagittal
                # RandFlipd(keys=keys, prob=prob, spatial_axis=1, allow_missing_keys=True), # Flip along coronal
                # RandFlipd(keys=keys, prob=prob, spatial_axis=2, allow_missing_keys=True), # Flip along axial
                RandZoomd(keys=keys,
                prob=prob,
                min_zoom=1.0,
                max_zoom=1.5,
                mode=("bilinear", "nearest"),
                padding_mode="reflection",
                allow_missing_keys=True,)
                # RandRotate90d(keys=keys, prob=prob, max_k=3, spatial_axes=(0, 1)),
            ]
            xforms += [RandomOrder([
            RandGaussianSmoothd(keys="image", sigma_x=(0.5, 2), sigma_y=(0.5, 2), prob=prob_intensity_appearance),
            RandScaleIntensityd(keys="image", factors=0.5, prob=prob_intensity_appearance),
            RandAdjustContrastd(keys="image", gamma=(0.5, 2), prob=prob_intensity_appearance),
            RandShiftIntensityd(keys="image", offsets=(-0.1, 0.1), prob=prob_intensity_appearance),
            # RandHistogramShiftd(keys="image", num_control_points=5, prob=1), 
            # RandGaussianSharpend(keys="image", prob=prob_intensity_appearance)
            ])]
            xforms += [RandGaussianNoised(keys="image", std=0.1, prob=0.5)]
        xforms += self._get_intensity_scaler()
            # For validation, we don't apply random transforms.
            # We return the whole pre-processed volume for sliding window inference.
        return Compose(xforms)

    def _get_strong_transforms(self):
        """Strong augmentations for contrastive learning."""
        keys = ["image", "label"]
        return Compose([
             # --- FEATURE: Smart Patch Sampling ---
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=self.patch_size,
                pos=0.8, neg=0.2,
                num_samples=self.num_samples_per_volume,
                image_key="image",
                image_threshold=0,
            ),
            # --- FEATURE: Rich 3D Augmentations ---
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=keys, prob=0.5, max_k=3, spatial_axes=(0, 1)),
            RandAffined(
                keys=keys,
                prob=0.8,
                rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12), # ~15 degrees
                translate_range=(self.patch_size[0]*0.1, self.patch_size[1]*0.1, self.patch_size[2]*0.1),
                scale_range=(0.8, 1.2),
                mode=("bilinear", "nearest"),
                padding_mode="reflection",
            ),
            # --- Intensity and Appearance Augmentations ---
            RandomOrder([
                RandGaussianSmoothd(keys="image", sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5), prob=0.3),
                RandScaleIntensityd(keys="image", factors=0.3, prob=0.3),
                RandAdjustContrastd(keys="image", gamma=(0.7, 1.5), prob=0.3),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
            ]),
            RandGaussianNoised(keys="image", std=0.02, prob=0.2),
        ])

    def __len__(self):
        # Each item in file_list corresponds to one 3D volume.
        # The sampler inside the transforms handles picking patches.
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load and pre-process the entire volume once
        if self.file_list[idx]["label"] == "N/A":
            del self.file_list[idx]["label"]
        base_data = self.base_transforms(self.file_list[idx])
        if not self.is_train:
            # For validation, return the whole pre-processed volume
            return {"image": base_data["image"].permute(0, 3, 1, 2), "label": base_data["label"].permute(0, 3, 1, 2).long()}

        if self.is_contrastive:
            # Generate two strongly augmented views from the same volume
            # We apply the transform to copies to ensure they are independent
            view1_list = self.strong_transforms(base_data.copy())
            view2_list = self.strong_transforms(base_data.copy())
            
            # RandCrop... returns a list of dictionaries, pick the first from each
            view1 = view1_list[0]
            view2 = view2_list[0]

            return {
                "image": view1["image"],
                "image2": view2["image"],
                "label": view1["label"].long(), # Label from the first view
            }
        else:
            # Generate weakly augmented views for standard supervised training
            processed_data = self.weak_transforms(base_data)
            # Here you can return one or multiple patches. For simplicity, we return the first.
            # To use all 'num_samples_per_volume', you would need a collate_fn that can handle lists.
            # processed_data = processed_list[0]
            
            return {
                "image": processed_data["image"].permute(0, 3, 1, 2),  # Change to (C, D, H, W) format
                "label": processed_data["label"].permute(0, 3, 1, 2).long() if "label" in processed_data else torch.tensor([]),
            }
    
# Example usage:
if __name__ == "__main__":
#     # Assuming hf_dataset is already defined and loaded
    patch_size = (512, 512)  # Example patch size
    name = "train_ct_gt"  # Example dataset name
    hf_dataset = load_dataset("./local_flare_loader.py", name=name, data_dir="/scratch/work/zhul2/data/FLARE-MedFM/FLARE-Task3-DomainAdaption", trust_remote_code=True)["train"]
    # dataset = OnTheFly2DDataset(hf_dataset, patch_size=patch_size, is_train=True, is_contrastive=True, has_label=True)
    dataset = OnTheFly2DDataset(
        hf_dataset,
        patch_size=patch_size, is_train=True, is_contrastive=True, has_label=True, slice_based=True, volume_type="ct"
    )
    # plot image, image2
    # visualize a batch of images
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    images = batch["image"]
    if "image2" not in batch:
        batch["image2"] = images
    images2 = batch["image2"]
    labels = batch["label"] if "label" in batch else torch.tensor([])
    print(torch.unique(labels))
    print("unique labels:", len(torch.unique(labels)))
    print(f"Batch size: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Image2 shape: {images2[0].shape}")
    print(f"Label shape: {labels[0].shape}")
    # Plot the first image and its corresponding label
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs[0, 0].imshow(images[0].squeeze().cpu().numpy(), cmap='gray')
    axs[0, 0].set_title("Image 1")
    axs[0, 1].imshow(images2[0].squeeze().cpu().numpy(), cmap='gray')
    axs[0, 1].set_title("Image 2")
    if labels.numel() > 0:
        axs[0, 2].imshow(labels[0].squeeze().cpu().numpy())
    axs[1, 0].imshow(images[1].squeeze().cpu().numpy(), cmap='gray')
    axs[1, 0].set_title("Image 3")
    axs[1, 1].imshow(images2[1].squeeze().cpu().numpy(), cmap='gray')
    axs[1, 1].set_title("Image 4")
    if labels.numel() > 0:
        axs[1, 2].imshow(labels[1].squeeze().cpu().numpy())
    plt.show()
    plt.savefig(f"{name}_example.png")
    # plot histogram of the two images
    plt.figure(figsize=(10, 5))
    plt.hist(images[0].cpu().numpy().flatten(), bins=100, alpha=0.5, label='Image 1')
    plt.hist(images[1].cpu().numpy().flatten(), bins=100, alpha=0.5, label='Image 2')
    plt.title("Histogram of Image Intensities")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    plt.savefig(f"{name}_histogram.png")
    # draw mri
    
    name = "train_mri_unlabeled"  # Example dataset name
    hf_dataset = load_dataset("./local_flare_loader.py", name=name, data_dir="/scratch/work/zhul2/data/FLARE-MedFM/FLARE-Task3-DomainAdaption", trust_remote_code=True)["train"]
    # dataset = OnTheFly2DDataset(hf_dataset, patch_size=patch_size, is_train=True, is_contrastive=True, has_label=True)
    dataset = OnTheFly2DDataset(
        hf_dataset,
        patch_size=patch_size, is_train=True, is_contrastive=True, has_label=True, slice_based=True, volume_type="mri"
    )
    # plot image, image2
    # visualize a batch of images
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
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
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs[0, 0].imshow(images[0].squeeze().cpu().numpy(), cmap='gray')
    axs[0, 0].set_title("Image 1")
    axs[0, 1].imshow(images2[0].squeeze().cpu().numpy(), cmap='gray')
    axs[0, 1].set_title("Image 2")
    if labels.numel() > 0:
        axs[0, 2].imshow(labels[0].squeeze().cpu().numpy())
    axs[1, 0].imshow(images[1].squeeze().cpu().numpy(), cmap='gray')
    axs[1, 0].set_title("Image 3")
    axs[1, 1].imshow(images2[1].squeeze().cpu().numpy(), cmap='gray')
    axs[1, 1].set_title("Image 4")
    if labels.numel() > 0:
        axs[1, 2].imshow(labels[1].squeeze().cpu().numpy())
    plt.show()
    plt.savefig(f"{name}_example.png")
    # plot histogram of the two images
    plt.figure(figsize=(10, 5))
    plt.hist(images[0].cpu().numpy().flatten(), bins=100, alpha=0.5, label='Image 1')
    plt.hist(images[1].cpu().numpy().flatten(), bins=100, alpha=0.5, label='Image 2')
    plt.title("Histogram of Image Intensities")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    plt.savefig(f"{name}_histogram.png")
    


# test 3d dataset
    # patch_size = (224, 192, 40)  # Example patch size
    # name = "train_mri_unlabeled"  # Example dataset name
    # hf_dataset = load_dataset("./local_flare_loader.py", name=name, data_dir="/scratch/work/zhul2/data/FLARE-MedFM/FLARE-Task3-DomainAdaption", trust_remote_code=True)["train"]
    # dataset = Flare3DPatchDataset(hf_dataset, patch_size=patch_size, is_train=True, is_contrastive=False, volume_type="mri", target_spacing=(1.0, 1.0, 1.0))
    # # visualize a batch of images
    # import matplotlib.pyplot as plt
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # batch = next(iter(dataloader))
    # images = batch["image"]
    # labels = batch["label"] if "label" in batch else torch.tensor([])
    # print("unique labels:", len(torch.unique(labels)))
    # print(f"Batch size: {len(images)}")
    # print(f"Image shape: {images.shape}")
    # print(f"Label shape: {labels.shape}")
    # # Plot the first image and its corresponding label, and plot the slices
    # fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    # axs[0, 0].imshow(images[0][0, 0, :, :].squeeze().cpu().numpy(), cmap='gray')
    # axs[0, 0].set_title("Image 1")
    # axs[0, 1].imshow(images[0][0, 1, :, :].squeeze().cpu().numpy(), cmap='gray')
    # axs[0, 1].set_title("Image 2")
    # if labels.numel() > 0:
    #     axs[0, 2].imshow(labels[0][0, 0, :, :].squeeze().cpu().numpy())
    # axs[1, 0].imshow(images[0][0, 0, :, :].squeeze().cpu().numpy(), cmap='gray')
    # axs[1, 0].set_title("Image 3")
    # axs[1, 1].imshow(images[0][0, 1, :, :].squeeze().cpu().numpy(), cmap='gray')
    # axs[1, 1].set_title("Image 4")
    # if labels.numel() > 0:
    #     axs[1, 2].imshow(labels[0][0, 0, :, :].squeeze().cpu().numpy())
    # plt.savefig(f"{name}_example.png")
    # # plot histogram of the two images
    # plt.figure(figsize=(10, 5))
    # plt.hist(images[0][0, 0, :, :].cpu().numpy().flatten(), bins=100, alpha=0.5, label='Image 1')
    # plt.hist(images[0][0, 1, :, :].cpu().numpy().flatten(), bins=100, alpha=0.5, label='Image 2')
    # plt.title("Histogram of Image Intensities")
    # plt.xlabel("Intensity")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.savefig(f"{name}_histogram.png")
