import torch
# from torch.utils.data import Dataset
from monai.data import Dataset
import SimpleITK as sitk
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
    RandRotate90d
)

from monai.data import NibabelReader

import torch
import random
from typing import Dict, Union, List, Any
from datasets import load_dataset
import numpy as np
# --- STEP 1: Create a simple, self-contained custom transform ---
    
class LoadSlice(MapTransform):
    """
    A simple MONAI transform to load a single 2D slice from a 3D NIfTI file.
    This is space-efficient as it doesn't require pre-processing.
    """
    def __init__(self, keys: list, axis: int = 2):
        """
        Args:
            keys: The keys in the data dictionary that are file paths to load.
            axis: The axis from which to extract the slice (0=sag, 1=cor, 2=ax).
        """
        super().__init__(keys)
        self.axis = axis

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data) # Create a copy to modify
        
        # We need a slice index. Let's get it from the 'image' metadata.
        # First, figure out the depth of the image volume.
        image_path = d['image']
        reader = sitk.ImageFileReader()
        reader.SetFileName(image_path)
        reader.ReadImageInformation()
        depth = reader.GetSize()[self.axis]
        
        # Pick a random slice index. This index will be used for all keys.
        slice_idx = random.randint(0, depth - 1)
        
        # Now, load the specific slice for each key in our list
        for key in self.keys:
            if key in d:
                filepath = d[key]
                
                # Load the full 3D image header and extract one slice
                img_sitk = sitk.ReadImage(filepath)
                
                size = list(img_sitk.GetSize())
                index = [0, 0, 0]
                index[self.axis] = slice_idx
                size[self.axis] = 1 # Extract a slice of thickness 1
        
                extractor = sitk.ExtractImageFilter()
                extractor.SetSize(size)
                extractor.SetIndex(index)
                img_slice_sitk = extractor.Execute(img_sitk)
                
                # Get pixel data and update the dictionary
                pixel_data = sitk.GetArrayFromImage(img_slice_sitk).squeeze(0) # Shape: (H, W)
                d[key] = pixel_data
        # # move d[key] to gpu cuda
        # for key in d.keys():
        #     if isinstance(d[key], np.ndarray):
        #         d[key] = torch.tensor(d[key], dtype=torch.float32).cuda()
        
                
        return d

class MinimalOnTheFly2DDataset(Dataset):
    """
    A minimal dataset that ONLY loads a 2D slice from disk and converts its type.
    All expensive augmentations are deferred to the GPU.
    """
    def __init__(self, hf_dataset):
        self.data_dicts = []
        for item in hf_dataset:
            if item.get('image_path') and item['image_path'] != "N/A":
                self.data_dicts.append({
                    "image": item['image_path'],
                    "label": item.get('label_path', None),
                    "label1": item.get('label_path1', None)
                })

        if not self.data_dicts:
            raise ValueError("hf_dataset did not yield any valid image paths.")
            
        # This is the entire transformation pipeline for the CPU.
        # It's fast because it does no heavy computation.
        self.cpu_transforms = Compose([
            LoadSlice(keys=["image", "label", "label1"]),
            EnsureChannelFirstd(keys=["image", "label", "label1"], allow_missing_keys=True, channel_dim="no_channel"),
            EnsureTyped(keys="image", dtype=torch.float32),
            EnsureTyped(keys=["label", "label1"], dtype=torch.int32, allow_missing_keys=True),
        ])

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        item_dict = self.data_dicts[idx].copy()

        # In training, randomly decide which pseudo-label to use (if available)
        label_path_to_use = None
        if item_dict["label"] != "N/A" and item_dict["label1"] != "N/A":
            # If training with two valid labels, randomly pick one's path
            chosen_key = random.choice(["label", "label1"])
            label_path_to_use = item_dict[chosen_key]
        else:
            # For validation or if only one label exists, use the primary 'label' if available
            label_path_to_use = item_dict.get("label", "N/A")
        clean_dict = {
                "image": item_dict["image"],
                "label": label_path_to_use
            }
        
        if clean_dict["label"] == "N/A":
        # If no label, we can't use this for supervised training/validation.
        # Here we choose to remove the key so transforms like RandCropByPosNegLabeld don't fail.
            del clean_dict["label"]
        
        # MONAI's Compose pipeline handles everything from here
        processed_data = self.cpu_transforms(clean_dict)
        if 'label' in processed_data:
            processed_data['label'] = processed_data['label'].long()
            
        return processed_data
# --- STEP 2: The Dataset class that uses our new transform ---

class OnTheFly2DDataset(Dataset):
    """
    An efficient 2D Dataset that loads slices on-the-fly without pre-processing.
    This version uses a simple custom MONAI transform, making it robust and bug-free.
    """
    def __init__(self, hf_dataset, patch_size=(224, 192), is_train=True, is_contrastive=False):
        self.is_train = is_train
        self.patch_size = patch_size
        self.is_contrastive = is_contrastive

        # The data for MONAI is just a list of dictionaries with file paths
        self.data_dicts = []
        for item in hf_dataset:
            # We only need valid pairs for the dataset
            if item.get('image_path') and item['image_path'] != "N/A":
                self.data_dicts.append({
                    "image": item['image_path'],
                    "label": item.get('label_path', 'N/A'),
                    "label1": item.get('label_path1', 'N/A')
                })

        if not self.data_dicts:
            raise ValueError("hf_dataset did not yield any valid image paths.")

        # The transformation pipeline now includes our custom loader
        self.transforms = self._get_transforms()
        
    def _get_transforms(self):
        # The keys we want our custom loader to process
        loading_keys = ["image", "label"]
        
        xforms = [
            # Our custom transform is the first step!
            LoadSlice(keys=loading_keys),
            # These transforms now operate on the 2D slice data in memory
            EnsureChannelFirstd(keys=loading_keys, channel_dim="no_channel"),
            EnsureTyped(keys="image", dtype=torch.float32),  # Ensure image is float32 and on GPU
            EnsureTyped(keys="label", dtype=torch.int8),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=self.patch_size),
        ]
        
        if self.is_train:
            # resize double the size of patch_size if image_size is smaller than patch_size
            xforms.extend([
                # Note: RandSpatialCropd will only operate on keys that exist in the dict
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.5,
                    max_k=3, # Max number of 90-degree rotations
                    spatial_axes=(0, 1), # Rotate in the X-Y plane
                ),
                RandAffined(
                    keys=["image", "label"],
                    prob=0.1,
                    rotate_range=(0.05,),          # 2D rotation
                    scale_range=(0.1,),             # 2D isotropic scaling
                    translate_range=(5, 5),         # 2D translation (H, W)
                    shear_range=(0.05,),            # 2D shear
                    spatial_size=self.patch_size,   # The target 2D patch size
                    mode=("bilinear", "nearest"),
                ),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
                # RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                RandGaussianNoised(keys="image", std=0.1, prob=0.01),
            ])
        
        xforms.append(NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
        return Compose(xforms)

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        # Get the initial dictionary of file paths
        item_dict = self.data_dicts[idx].copy()

        # In training, randomly decide which pseudo-label to use (if available)
        label_path_to_use = None
        if self.is_train and item_dict["label"] != "N/A" and item_dict["label1"] != "N/A":
            # If training with two valid labels, randomly pick one's path
            chosen_key = random.choice(["label", "label1"])
            label_path_to_use = item_dict[chosen_key]
        else:
            # For validation or if only one label exists, use the primary 'label' if available
            label_path_to_use = item_dict.get("label", "N/A")
        clean_dict = {
                "image": item_dict["image"],
                "label": label_path_to_use
            }
        if clean_dict["label"] == "N/A":
        # If no label, we can't use this for supervised training/validation.
        # Here we choose to remove the key so transforms like RandCropByPosNegLabeld don't fail.
            del clean_dict["label"]
        
        # MONAI's Compose pipeline handles everything from here
        processed_data = self.transforms(clean_dict)

        # For validation, we need to crop manually if needed, as Rand... transforms are off
        if not self.is_train:
            # Example of a center crop for validation
            cropper = RandSpatialCropd(keys=["image", "label"], roi_size=self.patch_size, random_size=False)
            processed_data = cropper(processed_data)

        if len(np.unique(processed_data["label"])) > 14:
            raise ValueError(f"Label has more than 14 classes: {np.unique(processed_data['label'])}. "
                             "Ensure labels are one-hot encoded or properly formatted.")
            
        return processed_data
    
    
class RandSliceSampling(torch.nn.Module):
    """
    Randomly selects a slice along the depth dimension from a 4D tensor (C, D, H, W).
    """
    def __init__(self, keys: List[str], dim: int = 1, prob: float = 1.0):
        """
        Args:
            keys: Keys in the data dictionary to apply the transform to.
            dim: The dimension along which to sample slices. For (C, D, H, W), dim=1 is depth.
            prob: Probability of applying the transform.
        """
        super().__init__()
        self.keys = keys
        self.dim = dim
        self.prob = prob

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.same_prob(self.prob):
            return data

        new_data = {}
        for key in data.keys():
            if key in self.keys:
                tensor = data[key]
                if tensor.ndim == 4: # Expecting (C, D, H, W)
                    depth_size = tensor.shape[self.dim]
                    if depth_size <= 0:
                        raise ValueError(f"Dimension {self.dim} has size {depth_size}, cannot sample.")
                    slice_index = random.randint(0, depth_size - 1)
                    # Select the slice along the specified dimension
                    sliced_tensor = torch.index_select(tensor, self.dim, torch.tensor(slice_index))
                    new_data[key] = sliced_tensor.squeeze(self.dim) 
                else:
                    # If not 4D, pass through unchanged (e.g., label if it's already processed)
                    new_data[key] = tensor
            else:
                new_data[key] = data[key] # Pass through other keys unchanged)

        return new_data

    def same_prob(self, p: float):
        """Helper to check if the transform should be applied based on probability."""
        return random.random() < p
    
    
    
class Flare2DPatchDataset(Dataset):
    """
    Robust 2D Dataset for the FLARE challenge using 2D patch sampling with random slice selection.
    Assumes 3D NIfTI files are loaded, and then a random slice is taken along the depth
    dimension to create 2D data.
    """
    def __init__(self, hf_dataset, patch_size=(224, 192), is_train=True, is_contrastive=False):
        self.file_list = []
        for item in hf_dataset:
            # Handle cases with potentially missing labels
            if item.get('label_path') and item['label_path'] != "N/A":
                self.file_list.append({"image": item['image_path'], "label": item['label_path']})
                self.keys = ["image", "label"]
                if item["label_path1"] != "N/A":
                    # If there's a second label, append it too
                    # add label1 to the last file_list
                    self.file_list[-1]["label1"] = item['label_path1']
                    self.keys.append("label1")

        if not self.file_list:
            raise ValueError("No valid image-label pairs found. Ensure your data has labels.")

        self.patch_size = patch_size # Should be (H, W)
        self.is_train = is_train
        self.is_contrastive = is_contrastive

        self.transforms = self.get_transforms()

    def get_transforms(self):
        """Builds and returns the MONAI transformation pipeline for 2D patches."""
        # if exist two labels, randomly select one
        if len(self.keys) > 2:
            label_value = random.choice(self.keys[1:])  # Randomly select one of the label keys
            self.keys = [self.keys[0], label_value]  # Keep only the image and the selected label
        # Initial transforms
        initial_transforms = [
            LoadImaged(keys=self.keys, reader=NibabelReader(to_gpu=True)),
            EnsureChannelFirstd(keys=self.keys, channel_dim="no_channel"), # Converts to (C, D, H, W)
            EnsureTyped(keys=self.keys),
            RandSliceSampling(keys=self.keys, dim=-1)
        ]

        if self.is_train:
            train_transforms = [
                # --- CHANGE ---
                RandSpatialCropd(
                    keys=self.keys,
                    roi_size=self.patch_size,
                ),
                # Flips are now 2D. spatial_axis=0 is H, spatial_axis=1 is W.
                RandFlipd(keys=self.keys, prob=0.5, spatial_axis=0), # Flip along H
                RandFlipd(keys=self.keys, prob=0.5, spatial_axis=1), # Flip along W
                
                RandGaussianNoised(keys="image", std=0.01, prob=0.1),
                
                # --- CHANGE ---
                # RandAffined parameters must now be 2D.
                RandAffined(
                    keys=self.keys,
                    prob=0.5,
                    rotate_range=(0.05,),          # 2D rotation
                    scale_range=(0.1,),             # 2D isotropic scaling
                    translate_range=(5, 5),         # 2D translation (H, W)
                    shear_range=(0.05,),            # 2D shear
                    spatial_size=self.patch_size,   # The target 2D patch size
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                
            ]
            return Compose(initial_transforms + train_transforms)

        else: # Validation transforms
            val_transforms = [
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
            return Compose(initial_transforms + val_transforms)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data_dict = self.transforms(self.file_list[idx])

        # After RandSliceSamplingd and RandCropByPosNegLabeld,
        # image and label should be (C, H, W) e.g., (1, 224, 192)
        image = data_dict['image']
        # check if label is a string, if so use label1
        if isinstance(data_dict["label"], str):
            label = data_dict['label1'].long()  # Ensure label is a single channel
        else:
            label = data_dict['label'].long()  # Ensure label is a single channel

        
        if self.is_contrastive:
            data_dict2 = self.transforms(self.file_list[idx]) # Re-run transforms for the second view
            image2 = data_dict2['image']
            label2 = data_dict2['label'].long()
            return image, image2, label, label2

        if not self.is_train:
            # For validation, return the processed slice along with path
            return {"image": image, "label": label, "path": self.file_list[idx]['image']}

        return {"image": image, "label": label}






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
    hf_dataset = load_dataset("./local_flare_loader.py", name="train_ct_pseudo", data_dir="/scratch/work/zhul2/data/FLARE-MedFM/FLARE-Task3-DomainAdaption", trust_remote_code=True)["train"]
    dataset = OnTheFly2DDataset(hf_dataset, patch_size=patch_size, is_train=True)
    print(f"Dataset length: {len(dataset)}")
    for i in range(len(dataset)):
        sample = dataset[i]
        # if len(torch.unique(sample['label'])) > 14:
        #     print(f"Sample {i} has more than 14 classes in label: {torch.unique(sample['label'])}")
        # Print the shapes of the image and label tensors
        if 'label' in sample:
            print(f"Sample {i} - Image shape: {sample['image'].shape}, Label shape: {sample['label'].shape}, Label unique values: {torch.unique(sample['label']).size()}")
        else:
            print(f"Sample {i} - Image shape: {sample['image'].shape}, No label available")
        # Uncomment to see the actual data
        # print(sample)

