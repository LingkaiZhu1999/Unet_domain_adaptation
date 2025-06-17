import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from datasets import load_dataset
from tqdm import tqdm
import time
import os
import numpy as np
import pandas as pd
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from itertools import cycle

# --- Project-specific imports ---
from unet import Unet_SegCLR  # Assumes this model exists and is adaptable
from loss import NTXentLoss
from loss import MultiClassDiceCELoss
from metrics import AverageMeter
from dataset import FlareSliceDataset, NiftiDataset # Use our new dataset class
from monai.metrics import DiceMetric

class SegCLR_FLARE:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.nt_xent_loss = NTXentLoss(self.device, 2 * self.args.batch_size, self.args.temperature, use_cosine_similarity=True)
        self.writer = SummaryWriter(log_dir=f'./models/{self.args.name}')

    def joint_train_on_source_and_target(self):
        # --- 1. Define Augmentations ---
        # Simplified for single-channel images
        train_transform = A.Compose([
        A.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0), p=1.0),    # MRI size is 512x512 PET size is 400x400
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.Compose([
                A.ToRGB(p=1.0),
                A.ColorJitter(brightness=0.6, contrast=0.2, p=1.0),
                A.ToGray(p=1.0),
            ])
        ], p=0.8),
        ],
        )

        # --- 2. Load Datasets using the FLARE Loader ---
        print("--- Loading FLARE Datasets ---")
        LOCAL_DATASET_PATH = "/mnt/asgard6/data/FLARE-MedFM/FLARE-Task3-DomainAdaption"
        
        # Source Domain: Labeled CT scans (GT + Pseudo)
        hf_source_gt = load_dataset("./local_flare_loader.py", name="train_ct_gt", data_dir=LOCAL_DATASET_PATH, trust_remote_code=True)["train"]
        hf_source_pseudo = load_dataset("./local_flare_loader.py", name="train_ct_pseudo_blackbean_flare22", data_dir=LOCAL_DATASET_PATH, trust_remote_code=True)["train"]
        hf_source = ConcatDataset([hf_source_gt, hf_source_pseudo])

        # Target Domain: Unlabeled MRI scans
        hf_target = load_dataset("./local_flare_loader.py", name="train_mri_unlabeled", data_dir=LOCAL_DATASET_PATH, trust_remote_code=True)["train"]

        # Validation Set: Labeled CT for supervised metric check
        hf_val_source = load_dataset("./local_flare_loader.py", name="validation_mri", data_dir=LOCAL_DATASET_PATH, trust_remote_code=True)["train"]

        # Create PyTorch Slice Datasets for training/contrastive validation
        train_source_dataset = FlareSliceDataset(hf_source, augmentation=train_transform, is_contrastive=True)
        train_target_dataset = FlareSliceDataset(hf_target, augmentation=train_transform, is_contrastive=True)
        val_source_dataset = FlareSliceDataset(hf_val_source, augmentation=train_transform, is_contrastive=True) # For contrastive validation

        # Create DataLoader for 3D validation
        val_source_3d_dataset = NiftiDataset(hf_val_source)

        # --- 3. Create DataLoaders ---
        train_source_loader = DataLoader(train_source_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        train_target_loader = DataLoader(train_target_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_source_loader = DataLoader(val_source_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        val_source_3d_loader = DataLoader(val_source_3d_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        # --- 4. Initialize Model, Loss, Optimizer ---
        model = Unet_SegCLR(in_channel=self.args.input_channel, out_channel=self.args.output_channel).to(self.device)
        # For multi-class segmentation (13 organs + 1 background)
        criterion = MultiClassDiceCELoss(num_classes=self.args.output_channel).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), self.args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=0)
        scaler = GradScaler(enabled=True)

        # --- 5. Training Loop ---
        best_val_dice = 0
        trigger = 0
        target_iterator = cycle(train_target_loader)

        for epoch in range(self.args.epochs):
            model.train()
            sup_loss_meter = AverageMeter()
            con_loss_meter = AverageMeter()

            for i, data_source in tqdm(enumerate(train_source_loader), total=len(train_source_loader)):
                image_source1, image_source2, label_source1, _ = data_source
                image_target1, image_target2, _, _ = next(target_iterator)

                image_source1, image_source2 = image_source1.to(self.device), image_source2.to(self.device)
                label_source1 = label_source1.to(self.device)
                image_target1, image_target2 = image_target1.to(self.device), image_target2.to(self.device)

                optimizer.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # Forward passes
                    z11, predict_source1 = model(image_source1)
                    z22, _ = model(image_source2)
                    z1, _ = model(image_target1)
                    z2, _ = model(image_target2)

                    # Supervised Loss on source domain
                    supervise_loss = criterion(predict_source1, label_source1)
                    
                    # Contrastive Loss (logic adapted from your code)
                    if self.args.contrastive_mode == 'inter_domain':
                        z_all1 = torch.cat((z1, z11), dim=0)
                        z_all2 = torch.cat((z2, z22), dim=0)
                        contrast_loss = self.nt_xent_loss(F.normalize(z_all1, dim=1), F.normalize(z_all2, dim=1))
                    elif self.args.contrastive_mode == 'within_domain':
                        loss1 = self.nt_xent_loss(F.normalize(z1, dim=1), F.normalize(z2, dim=1))
                        loss2 = self.nt_xent_loss(F.normalize(z11, dim=1), F.normalize(z22, dim=1))
                        contrast_loss = 0.5 * (loss1 + loss2)
                    else: # 'only_target_domain'
                        contrast_loss = self.nt_xent_loss(F.normalize(z1, dim=1), F.normalize(z2, dim=1))

                    total_loss = self.args.lam * supervise_loss + contrast_loss

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                sup_loss_meter.update(supervise_loss.item(), image_source1.size(0))
                con_loss_meter.update(contrast_loss.item(), image_source1.size(0))

            print(f'Training: Epoch [{epoch+1}/{self.args.epochs}] | Sup Loss: {sup_loss_meter.avg:.4f} | Con Loss: {con_loss_meter.avg:.4f}')
            self.writer.add_scalar("Loss/train_supervise", sup_loss_meter.avg, epoch)
            self.writer.add_scalar("Loss/train_contrastive", con_loss_meter.avg, epoch)
            
            if (epoch + 1) % self.args.validate_frequency == 0:
                # --- Validation Step ---
                val_dice = self.validate_slice_dice(model, val_source_loader)
                self.writer.add_scalar("Metric/val_dice_2d", val_dice, epoch)
                print(f'Validation: Epoch [{epoch+1}/{self.args.epochs}] | 2D Slice Dice: {val_dice:.4f}')

                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    trigger = 0
                    torch.save(model.state_dict(), os.path.join(f'./models/{self.args.name}', 'best_val_dice_model.pt'))
                    print("=> Saving best model")
                else:
                    trigger += 1

            if self.args.early_stop and trigger >= self.args.early_stop:
                print("=> Early stopping")
                break
            
            if epoch >= self.args.warm_up:
                scheduler.step()
        
        self.writer.close()

    def validate_slice_dice(self, model, val_loader):
        """ Validates by computing Dice on 2D slices. """
        model.eval()
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # The validation dataset is not contrastive
                # A simple FlareSliceDataset with is_contrastive=False would be better
                # but we can just use the first image from the contrastive set
                image, _, label, _ = batch
                image, label = image.to(self.device), label.to(self.device)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _, prediction = model(image)
                
                # Convert prediction to one-hot format for DiceMetric
                pred_one_hot = F.one_hot(prediction.argmax(dim=1), num_classes=self.args.output_channel).permute(0, 3, 1, 2)
                label_one_hot = F.one_hot(label, num_classes=self.args.output_channel).permute(0, 3, 1, 2)
                
                dice_metric(y_pred=pred_one_hot, y=label_one_hot)
        
        mean_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        model.train()
        return mean_dice


if __name__ == '__main__':
    trainer = SegCLR_FLARE(args)
    trainer.joint_train_on_source_and_target()