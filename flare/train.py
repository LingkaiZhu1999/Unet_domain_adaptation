# File: train.py
import argparse
import os
import time
from collections import OrderedDict
from pathlib import Path

import albumentations as A
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from monai.data import decollate_batch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- Project-specific imports ---
from dataset import FlareSliceDataset
from datasets import load_dataset
from loss import MultiClassDiceCELoss
from metrics import AverageMeter  # Assuming you have this helper class
from unet import Unet
from utils import save_validation_results, mask_label_process  # Assuming you have this utility function

# --- 1. Configuration and Argument Parsing ---

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Training for FLARE Segmentation")
    # --- Data and Model Configuration ---
    parser.add_argument('--name', default='', help='Model name, auto-generated if left empty')
    parser.add_argument('--data_dir', type=Path, required=True, help='Path to the root FLARE dataset directory.')
    parser.add_argument('--train_domain', default="train_ct_gt", help='Training set name (e.g., "train_ct_gt").')
    parser.add_argument('--val_domain', default="validation_mri", help='Validation set name.')
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', default=100, type=int, help='Number of total epochs to run.')
    parser.add_argument('--early_stop', default=None, type=int, help='Early stopping patience.')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Mini-batch size.')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate.')
    
    # --- System and Reproducibility ---
    parser.add_argument('--device', default='cuda:0', help='Device to use for training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for DataLoader.')
    args = parser.parse_args()

    # Determine output channels based on the validation domain
    if 'mri' in args.val_domain or 'ct' in args.val_domain:
        args.output_channel = 14
    elif 'pet' in args.val_domain:
        args.output_channel = 5
    else:
        raise ValueError(f"Cannot determine output channels for validation domain: {args.val_domain}")

    if not args.name:
        args.name = f"unet_{args.train_domain}_on_{args.val_domain}_bs{args.batch_size}_lr{args.lr}"

    return args

# --- 2. Training and Validation Epoch Functions ---

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    loss_meter = AverageMeter()
    
    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_meter.update(loss.item(), images.size(0))
        
    return OrderedDict([('loss', loss_meter.avg)])

def validate_epoch(model, loader, dice_metric, device):
    model.eval()
    dice_metric.reset()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            with autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
            
            # outputs = torch.argmax(outputs, keepdim=True) # Get the predicted class)
            # outputs = mask_label_process(outputs)  # Process the outputs to match label format
            outputs = (torch.sigmoid(outputs) > 0.5).int()

            # save_validation_results(images, labels, outputs, 1, Path('./validation_results'))
            dice_metric(y_pred=outputs, y=labels)
            
    # Aggregate the final metric
    val_dice = dice_metric.aggregate(reduction="mean_batch").mean().item()
    return OrderedDict([('val_dice', val_dice)])

# --- 3. Main Training Function ---

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # --- Setup Directories and Seeds ---
    model_dir = Path('./models') / args.name
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(model_dir))
    
    print('--- Config ---')
    for key, value in vars(args).items():
        print(f'{key:<15}: {value}')
    print('--------------')

    if args.seed is not None:
        torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # --- Data Loading and Augmentations ---
    train_transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
    ])
    
    print(f"Loading training data: {args.train_domain}")
    hf_train = load_dataset("./local_flare_loader.py", name=args.train_domain, data_dir=str(args.data_dir), trust_remote_code=True)["train"]
    train_dataset = FlareSliceDataset(hf_train, output_size=(512, 512), augmentation=train_transform)
    
    print(f"Loading validation data: {args.val_domain}")
    hf_val = load_dataset("./local_flare_loader.py", name=args.val_domain, data_dir=str(args.data_dir), trust_remote_code=True)["train"]
    val_dataset = FlareSliceDataset(hf_val, output_size=(512, 512), augmentation=None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    # --- Model, Loss, Optimizer, and Metrics ---
    model = Unet(in_channels=1, out_channels=args.output_channel).to(device)

    print("Compiling model (PyTorch 2.0+)...")
    try:
        model = torch.compile(model)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Model compilation failed, continuing without it. Error: {e}")

    criterion = MultiClassDiceCELoss(num_classes=args.output_channel)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, fused=True)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # lr = init_lr * ( 1 - epoch / args.epochs )^ 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9)
    scaler = GradScaler()

    # --- CRITICAL FIX: Use the correct DiceMetric for validation ---
    # dice_loss = DiceLoss(include_background=True)
    
    # dice_metric = LossMetric(loss_fn=dice_loss)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    # Define post-processing transforms for validation metric

    # --- Training Loop ---
    log_path = model_dir / 'log.csv'
    with open(log_path, 'w') as f:
        f.write('epoch,lr,loss,val_dice\n')
        
    best_dice = 0.0
    best_train_loss = float('inf')
    trigger = 0
    epoch_progress = tqdm(range(1, args.epochs + 1), desc="Training Progress", unit="epoch")
    for epoch in epoch_progress:
        
        train_log = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_log = validate_epoch(model, val_loader, dice_metric, device)
        
        scheduler.step()

        epoch_progress.set_postfix(loss=train_log['loss'], val_dice=val_log['val_dice'])

        # Logging
        writer.add_scalar("Loss/train", train_log['loss'], epoch)
        writer.add_scalar("Dice/val", val_log['val_dice'], epoch)
        writer.add_scalar("Meta/lr", optimizer.param_groups[0]['lr'], epoch)
        
        with open(log_path, 'a') as f:
            log_data = [epoch, optimizer.param_groups[0]['lr'], train_log['loss'], val_log['val_dice']]
            f.write(','.join(map(str, log_data)) + '\n')

        # Model Checkpointing
        trigger += 1
        if train_log['loss'] < best_train_loss or epoch == 1:
            # Save the uncompiled model state dict for portability
            # Access the original model via ._orig_mod if compiled
            state_dict_to_save = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save(state_dict_to_save, model_dir / 'best_model.pt')

            best_train_loss = train_log['loss']
            # print(f"=> Saved best model with Train Loss: {best_train_loss:.4f}")
            epoch_progress.set_description(f"Best Train Loss: {best_train_loss:.4f}")
            trigger = 0

        if args.early_stop and trigger >= args.early_stop:
            print(f"=> Early stopping after {args.early_stop} epochs of no improvement.")
            break
            
    writer.close()
    print(f"\nTraining finished. Best validation Dice: {best_dice:.4f}")

if __name__ == '__main__':
    main()