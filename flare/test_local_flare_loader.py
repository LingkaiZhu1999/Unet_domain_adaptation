# In a Jupyter Notebook cell

import torch
from datasets import load_dataset

from dataset import NiftiDataset

# --- 1. CONFIGURE YOUR VISUALIZATION ---

# Path to your local dataset
LOCAL_DATASET_PATH = "/mnt/asgard6/data/FLARE-MedFM/FLARE-Task3-DomainAdaption/train_CT_gt_label"

# Choose which dataset split to view. Options:
# 'train_ct_gt', 'train_ct_pseudo_aladdin', 'train_ct_pseudo_blackbean',
# 'validation_mri', 'validation_pet', 'train_mri_unlabeled', 'train_pet_unlabeled'
DATASET_CONFIG_NAME = 'train_ct_gt' 

# Choose which sample from that split to visualize (e.g., the 10th patient)
SAMPLE_INDEX = 200

# In a Jupyter Notebook cell

print(f"--- Loading configuration: '{DATASET_CONFIG_NAME}' ---")
print(f"--- Visualizing sample index: {SAMPLE_INDEX} ---")

# Load the Hugging Face dataset object (contains file paths)
hf_dataset = load_dataset(
    "./local_flare_loader.py", 
    name=DATASET_CONFIG_NAME, 
    data_dir=LOCAL_DATASET_PATH, 
    trust_remote_code=True
)["train"]

print(len(hf_dataset), "samples found in the dataset.")
# Create our PyTorch NiftiDataset
pytorch_dataset = NiftiDataset(hf_dataset)

# Get the specific sample
sample = pytorch_dataset[SAMPLE_INDEX]

# Move tensors to CPU and convert to NumPy for plotting
# The .squeeze(0) removes the channel dimension (1, D, H, W) -> (D, H, W)
image_3d = sample['image'].cpu().numpy().squeeze(0)
label_3d = sample['label'].cpu().numpy()

# Check if the sample has a real label or is unlabeled
has_label = sample['label'].min() != -1

print(f"\nImage shape: {image_3d.shape}")
print(f"Label shape: {label_3d.shape}")
print(f"Dataset has labels: {has_label}")

print(hf_dataset[100])