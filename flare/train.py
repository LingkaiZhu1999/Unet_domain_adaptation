# File: train_domain_adaptation.py

import torch
from torch.utils.data import DataLoader, ConcatDataset
from datasets import load_dataset
from itertools import cycle

# --- Import our custom Dataset class ---
from dataset import NiftiDataset

# --- 1. CONFIGURE THIS PATH ---
LOCAL_DATASET_PATH = "/mnt/asgard6/data/FLARE-MedFM/FLARE-Task3-DomainAdaption"
CT_GT_DATASET_PATH = f"{LOCAL_DATASET_PATH}/train_CT_gt_label"
PSEUDO_CT_GT_DATASET_PATH = f"{LOCAL_DATASET_PATH}/train_CT_pseudolabel"
MRI_DATASET_PATH = f"{LOCAL_DATASET_PATH}/train_MRI_unlabeled"
MRI_VAL_DATASET_PATH = f"{LOCAL_DATASET_PATH}/validation"
# -----------------------------


# --- 2. SETUP GPU DEVICE ---
# This is the standard way to select the GPU if available, otherwise fallback to CPU.
# This makes your code portable.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")
# ---------------------------


# --- 3. Load Data Components (No changes here) ---
print("--- Loading all data components for the All Data Track (CT -> MRI) ---")
source_gt_hf = load_dataset("./local_flare_loader.py", name="train_ct_gt", data_dir=CT_GT_DATASET_PATH, trust_remote_code=True)["train"]
source_pseudo_hf = load_dataset("./local_flare_loader.py", name="train_ct_pseudo_blackbean_flare22", data_dir=PSEUDO_CT_GT_DATASET_PATH, trust_remote_code=True)["train"]
target_mri_hf = load_dataset("./local_flare_loader.py", name="train_mri_unlabeled", data_dir=MRI_DATASET_PATH, trust_remote_code=True)["train"]
val_mri_hf = load_dataset("./local_flare_loader.py", name="validation_mri", data_dir=MRI_VAL_DATASET_PATH, trust_remote_code=True)["train"]
print("\n--- Data Loading Summary ---")
print(f"Source (Ground Truth CT): {len(source_gt_hf)} samples")
print(f"Source (Pseudo-labeled CT): {len(source_pseudo_hf)} samples")
print(f"Target (Unlabeled MRI): {len(target_mri_hf)} samples")
print(f"Validation (MRI): {len(val_mri_hf)} samples")

# --- 4. Create PyTorch Datasets and DataLoaders (No changes here) ---
source_gt_pytorch = NiftiDataset(source_gt_hf)
source_pseudo_pytorch = NiftiDataset(source_pseudo_hf)
target_mri_pytorch = NiftiDataset(target_mri_hf)
val_mri_pytorch = NiftiDataset(val_mri_hf)

source_ct_pytorch = ConcatDataset([source_gt_pytorch, source_pseudo_pytorch])
print(f"Total source samples (GT + Pseudo): {len(source_ct_pytorch)}")

# For GPU usage, enabling `pin_memory=True` can speed up CPU-to-GPU data transfers.
# It works best when `num_workers > 0`.
source_loader = DataLoader(source_ct_pytorch, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
target_loader = DataLoader(target_mri_pytorch, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_mri_pytorch, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# --- 5. The Domain Adaptation Training Loop with GPU ---
print("\n--- Starting Simulated Domain Adaptation Training Loop on GPU ---")

# (You would define your model and optimizer here)
# model = YourUNetModel().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

target_iterator = cycle(target_loader)
num_epochs = 1
num_batches_to_run = len(source_loader)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    # model.train() # Set the model to training mode
    for i, source_batch in enumerate(source_loader):
        if i >= num_batches_to_run:
            break
            
        target_batch = next(target_iterator)

        # --- MOVE DATA TO GPU ---
        # Get tensors from the batch and move them to the selected device
        source_image = source_batch['image'].to(device)
        source_label = source_batch['label'].to(device)
        target_image = target_batch['image'].to(device)
        # ------------------------

        # --- FAKE MODEL AND LOSS CALCULATION ON GPU ---
        # From this point on, all operations are on the GPU
        
        # optimizer.zero_grad()
        
        # 1. Supervised loss on the source domain (CT)
        # source_prediction = model(source_image)
        # segmentation_loss = some_loss_function(source_prediction, source_label)

        # 2. Unsupervised domain adaptation loss using both source and target
        # target_prediction = model(target_image)
        # adaptation_loss = some_adaptation_module(source_prediction, target_prediction)
        
        # 3. Combine losses and backpropagate
        # total_loss = segmentation_loss + (lambda_adaptation * adaptation_loss)
        # total_loss.backward()
        # optimizer.step()
        
        if (i + 1) % 1 == 0:
            print(f"  Batch {i+1}/{num_batches_to_run} processed...")
            # You can check the device of the tensors to confirm they are on the GPU
            print(f"    Source Image Device: {source_image.device}")
            print(f"    Source Label Device: {source_label.device}")
            print(f"    Target Image Device: {target_image.device}")


print("\n--- Training Simulation Finished ---")