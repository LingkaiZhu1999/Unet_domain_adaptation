# In your train.py or Jupyter Notebook

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import the new dataset class
from dataset import FlareSliceDataset 
from dataset import NiftiDataset
from torch.utils.data import ConcatDataset
from itertools import cycle
from tqdm import tqdm
# --- 1. Define an Augmentation Pipeline (using albumentations) ---
# This is a sample pipeline, you can customize it
aug_pipeline = None

# --- 2. Load the data using your loader ---
LOCAL_DATASET_PATH = "/mnt/asgard6/data/FLARE-MedFM/FLARE-Task3-DomainAdaption"
        
# Source Domain: Labeled CT scans (GT + Pseudo)
hf_source_gt = load_dataset("./local_flare_loader.py", name="train_ct_gt", data_dir=LOCAL_DATASET_PATH, trust_remote_code=True)["train"]
hf_source_pseudo = load_dataset("./local_flare_loader.py", name="train_ct_pseudo_blackbean_flare22", data_dir=LOCAL_DATASET_PATH, trust_remote_code=True)["train"]
# hf_source = ConcatDataset([hf_source_gt, hf_source_pseudo])
hf_source = hf_source_pseudo  # For simplicity, using only GT data here 
# check hf_source_pseudo

# Target Domain: Unlabeled MRI scans
# hf_target = load_dataset("./local_flare_loader.py", name="train_mri_unlabeled", data_dir=LOCAL_DATASET_PATH, trust_remote_code=True)["train"]

# Validation Set: Labeled CT for supervised metric check
# hf_val_source = load_dataset("./local_flare_loader.py", name="validation_mri", data_dir=LOCAL_DATASET_PATH, trust_remote_code=True)["train"]

# Create PyTorch Slice Datasets for training/contrastive validation
train_source_dataset = FlareSliceDataset(hf_source, augmentation=aug_pipeline, is_contrastive=True)
# train_target_dataset = FlareSliceDataset(hf_target, augmentation=aug_pipeline, is_contrastive=True)
# val_source_dataset = FlareSliceDataset(hf_val_source, augmentation=aug_pipeline, is_contrastive=True) # For contrastive validation

# Create DataLoader for 3D validation
# val_source_3d_dataset = NiftiDataset(hf_val_source)

# --- 3. Create DataLoaders ---
train_source_loader = DataLoader(train_source_dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
# train_target_loader = DataLoader(train_target_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
# val_source_loader = DataLoader(val_source_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)


# # --- 4. Use with a DataLoader ---
# print("--- Testing Contrastive DataLoader ---")
# contrastive_loader = DataLoader(train_source_dataset, batch_size=4, shuffle=True)
# image1, image2, label1, label2 = next(iter(contrastive_loader))
# print(f"Image 1 batch shape: {image1.shape}") # e.g., [4, 1, 256, 256]
# print(f"Label 1 batch shape: {label1.shape}") # e.g., [4, 256, 256]
# # get the unique ints in label1

# print("\n--- Testing Supervised DataLoader ---")
# supervised_loader = DataLoader(train_source_dataset, batch_size=4, shuffle=True)
# batch = next(iter(supervised_loader))
# print(f"Image batch shape: {batch['image'].shape}")
# print(f"Label batch shape: {batch['label'].shape}")

# print("\n--- Testing Validation DataLoader ---")
# val_loader = DataLoader(val_source_dataset, batch_size=4, shuffle=False)
# val_batch = next(iter(val_loader))
# print(f"Validation Image batch shape: {val_batch['image'].shape}")
# print(f"Validation Label batch shape: {val_batch['label'].shape}")

# for i, (image1, image2, label1, label2) in enumerate(train_source_loader):
#     print(f"Batch {i}:")
#     print(f"Image 1 shape: {image1.shape}")  # e.g., [4, 1, 256, 256]
#     print(f"Image 2 shape: {image2.shape}")  # e.g., [4, 1, 256, 256]
#     print(f"Label 1 shape: {label1.shape}")  # e.g., [4, 256, 256]
#     print(f"Label 2 shape: {label2.shape}")  # e.g., [4, 256, 256]

# load the data between 436 and 440
for i in range(441, 446):
    item = train_source_dataset[i]
    print(f"Item {i}:")
    print(f"Image 1 shape: {item[0].shape}")  # e.g., [1, 256, 256]
    print(f"Image 2 shape: {item[1].shape}")  # e.g., [1, 256, 256]
    print(f"Label 1 shape: {item[2].shape}")  # e.g., [256, 256]
    print(f"Label 2 shape: {item[3].shape}")  # e.g., [256, 256]

# loop over without using dataloder to check all the data having the same shape
# from dataset import FlareSliceDataset, NiftiDataset  # Ensure this is the correct import path
# for i in range(len(train_source_dataset)):
#     item = train_source_dataset[i]
#     print(i, item[0].shape, item[1].shape, item[2].shape, item[3].shape)  # Check image and label shapes
