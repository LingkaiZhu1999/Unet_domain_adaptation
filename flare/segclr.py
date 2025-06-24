# File: segclr_train.py
import argparse
from collections import OrderedDict
from pathlib import Path
from itertools import cycle

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from monai.metrics import DiceMetric
from monai.transforms import Compose, AsDiscrete
from monai.data import DataLoader, CacheDataset
from monai.data import decollate_batch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

# --- Project-specific imports ---
from datasets import load_dataset, concatenate_datasets
from loss import MultiClassDiceCELoss, NTXentLoss
from metrics import AverageMeter
from unet import Unet_SegCLR # The model that outputs features and logits
from utils import save_validation_results

# --- 1. Argument Parsing (Combined from both scripts) ---

def parse_args():
    parser = argparse.ArgumentParser(description="SegCLR Training for Domain Adaptation")
    # --- Data and Model ---
    parser.add_argument('--name', default='', help='Experiment name')
    parser.add_argument('--data_dir', type=Path, required=True, help='Root FLARE dataset directory')
    parser.add_argument('--cache_dir', type=Path, default='./.monai_cache', help='Directory for MONAI PersistentDataset cache')
    parser.add_argument('--source_domain1', default="train_ct_gt", help='Labeled source domain')
    parser.add_argument('--source_domain2', default="train_ct_pseudo", help='Second labeled source domain')
    parser.add_argument('--target_domain', default="train_mri_unlabeled", help='Unlabeled target domain')
    parser.add_argument('--val_domain', default="validation_mri", help='Validation domain')
    parser.add_argument('--output_channel', default=14, type=int, help='Number of output segmentation classes')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate for AdamW')
    parser.add_argument('--validate_every', default=2, type=int)

    # --- SegCLR Specific Hyperparameters ---
    parser.add_argument('--lam', default=1.0, type=float, help='Weight for the supervised loss')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature for NT-Xent loss')
    parser.add_argument('--contrastive_mode', default='within_domain', choices=['inter_domain', 'within_domain', 'only_target_domain'])

    # --- System and Reproducibility ---
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_workers', default=64, type=int)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    if not args.name:
        args.name = f"SegCLR_{args.source_domain1}_{args.source_domain2}_to_{args.target_domain}_lam{args.lam}_mode_{args.contrastive_mode}_lr{args.lr}_epochs{args.epochs}_{args.seed}"
    return args

# --- 2. Training and Validation Functions (Adapted for SegCLR) ---

def train_epoch_segclr(model, source_loader, target_iterator, optimizer, sup_criterion, con_criterion, scaler, device, args):
    model.train()
    sup_loss_meter = AverageMeter()
    con_loss_meter = AverageMeter()

    for i, source_batch in enumerate(source_loader):
        # --- Get data from both domains ---
        target_batch = next(target_iterator)
        
        s_img1 = source_batch['image'].to(device, non_blocking=True)
        s_img2 = source_batch['image2'].to(device, non_blocking=True)
        s_label = source_batch['label'].to(device, non_blocking=True)
        
        t_img1 = target_batch['image'].to(device, non_blocking=True)
        t_img2 = target_batch['image2'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, dtype=torch.float16):
            # --- Forward passes for all 4 images ---
            zs1, logits_s1 = model(s_img1) # Supervised + Contrastive
            zs2, _         = model(s_img2) # Contrastive only
            zt1, _         = model(t_img1) # Contrastive only
            zt2, _         = model(t_img2) # Contrastive only

            # --- Loss Calculation ---
            # 1. Supervised Loss (on source domain, first view)

            supervise_loss = sup_criterion(logits_s1, s_label)
            
            # 2. Contrastive Loss (based on selected mode)

            if args.contrastive_mode == 'inter_domain':
                z_all1 = torch.cat((zs1, zt1), dim=0)
                z_all2 = torch.cat((zs2, zt2), dim=0)
                z_all1 = F.normalize(z_all1, dim=1)
                z_all2 = F.normalize(z_all2, dim=1)
                contrast_loss = con_criterion(z_all1, z_all2)
            elif args.contrastive_mode == 'within_domain':
                zs1, zs2 = F.normalize(zs1, dim=1), F.normalize(zs2, dim=1)
                zt1, zt2 = F.normalize(zt1, dim=1), F.normalize(zt2, dim=1)
                # Contrastive loss for both source and target
                loss_s = con_criterion(zs1, zs2)
                loss_t = con_criterion(zt1, zt2)
                contrast_loss = 0.5 * (loss_s + loss_t)
            else: # 'only_target_domain'
                contrast_loss = con_criterion(zt1, zt2)
           
            # 3. Total Loss
            total_loss = args.lam * supervise_loss + contrast_loss
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sup_loss_meter.update(supervise_loss.item(), s_img1.size(0))
        con_loss_meter.update(contrast_loss.item(), t_img1.size(0))
        
    return OrderedDict([('sup_loss', sup_loss_meter.avg), ('con_loss', con_loss_meter.avg)])

def validate_epoch(model, loader, criterion, dice_metric, device, post_trans, epoch=0, save_path=""):
    model.eval()
    dice_metric.reset()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            with autocast(device_type=device.type, dtype=torch.float16):
                _, outputs = model(images) # We only need the segmentation logits for validation
                loss = criterion(outputs, labels)
            loss_meter.update(loss.item(), images.size(0))

            processed_outputs = [post_trans(output_item) for output_item in decollate_batch(outputs)]
            dice_metric(y_pred=processed_outputs, y=decollate_batch(labels))
    
    val_dice_dict = dice_metric.aggregate() # aggregate() now returns a dict per class
    # To get the mean dice across all (non-background) classes:
    val_dice = val_dice_dict.mean().item()
    save_validation_results(
        images, labels, torch.stack(processed_outputs), epoch,
        output_dir=save_path
    )
    return OrderedDict([('val_loss', loss_meter.avg), ('val_dice', val_dice),  ('val_dice_per_class', val_dice_dict.cpu().numpy()) # For detailed logging
    ])

# --- 3. Main Training Function ---

def main():
    args = parse_args()
    model_dir = Path('./models') / args.name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / 'validation_results'
    save_path.mkdir(parents=True, exist_ok=True)
    wandb.init(
        project="SegCLR_Domain_Adaptation",
        name=args.name,
        config=args,
        sync_tensorboard=True
    )
    device = torch.device(args.device)

    # --- Select appropriate 2D or 3D Dataset ---
    # For this example, we'll hardcode the 3D dataset. You can add the 2D logic back if needed.
    if True:
        print("Using 2D model.")
        from dataset import OnTheFly2DDataset as FlarePatchDataset
        PATCH_SIZE_TRAIN = (480, 480)
        PATCH_SIZE_VAL = (480, 480)  

    # --- Setup Directories and Seeds ---

    writer = SummaryWriter(log_dir=str(model_dir))
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # --- Data Loading using MONAI PersistentDataset ---
    print("--- Loading & Caching Datasets ---")
    hf_source_gt = load_dataset("./local_flare_loader.py", name=args.source_domain1, data_dir=str(args.data_dir), trust_remote_code=True)["train"]
    hf_source_pseudo = load_dataset("./local_flare_loader.py", name=args.source_domain2, data_dir=str(args.data_dir), trust_remote_code=True)["train"]
    hf_source = concatenate_datasets([hf_source_gt, hf_source_pseudo])  # Combine both source domains
    # hf_source = hf_source_pseudo  # For now, we only use the ground truth source domain
    hf_target = load_dataset("./local_flare_loader.py", name=args.target_domain, data_dir=str(args.data_dir), trust_remote_code=True)["train"]
    hf_val = load_dataset("./local_flare_loader.py", name=args.val_domain, data_dir=str(args.data_dir), trust_remote_code=True)["train"]

    # Datasets for contrastive training
    source_dataset = FlarePatchDataset(hf_source, patch_size=PATCH_SIZE_TRAIN, is_train=True, is_contrastive=True, has_label=True)
    target_dataset = FlarePatchDataset(hf_target, patch_size=PATCH_SIZE_TRAIN, is_train=True, is_contrastive=True, has_label=False)

    # Dataset for standard validation
    val_dataset = FlarePatchDataset(hf_val, patch_size=PATCH_SIZE_VAL, is_train=False, is_contrastive=False, has_label=True)

    source_cached_dataset = CacheDataset(data=source_dataset, cache_rate=1., num_workers=args.num_workers)
    target_cached_dataset = CacheDataset(data=target_dataset, cache_rate=1., num_workers=args.num_workers)
    val_cached_dataset = CacheDataset(data=val_dataset, cache_rate=1., num_workers=args.num_workers)

    # DataLoaders
    source_loader = DataLoader(source_cached_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    target_loader = DataLoader(target_cached_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_cached_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    target_iterator = cycle(target_loader)
    
    # --- Model, Losses, Optimizer ---
    model = Unet_SegCLR(in_channels=1, out_channels=args.output_channel).to(device)
    # limit of compile to be 64
    torch._dynamo.config.cache_size_limit = 64
    model = torch.compile(model, mode="max-autotune")
    
    sup_criterion = MultiClassDiceCELoss(num_classes=args.output_channel).to(device)
    con_criterion = NTXentLoss(device, args.batch_size, args.temperature, use_cosine_similarity=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, fused=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # --- Validation Metrics Setup ---
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    post_trans = Compose([AsDiscrete(argmax=True, to_onehot=args.output_channel)])

    # --- Main Training Loop ---
    best_val_dice = 0.0
    epoch_progress = tqdm(range(1, args.epochs + 1), desc="Training Progress", unit="epoch")
    for epoch in epoch_progress:
        
        train_log = train_epoch_segclr(model, source_loader, target_iterator, optimizer, sup_criterion, con_criterion, scaler, device, args)
        epoch_progress.set_postfix({
            'Epoch': epoch,
            'Sup Loss': train_log['sup_loss'],
            'Con Loss': train_log['con_loss'],
            'Total Loss': train_log['sup_loss'] * args.lam + train_log['con_loss'],
            'LR': optimizer.param_groups[0]['lr']
        })
        # Log training losses
        writer.add_scalar("Loss/Supervised", train_log['sup_loss'], epoch)
        writer.add_scalar("Loss/Contrastive", train_log['con_loss'], epoch)

        scheduler.step()
        writer.add_scalar("Meta/LR", optimizer.param_groups[0]['lr'], epoch)
        
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            val_log = validate_epoch(model, val_loader, sup_criterion, dice_metric, device, post_trans, epoch, save_path)
            epoch_progress.set_postfix({
                'Val Dice': val_log['val_dice'],
                'Val CEDiceLoss': val_log['val_loss']
            })
            # Log validation metrics
            writer.add_scalar("Dice/Validation_Average", val_log['val_dice'], epoch)
            writer.add_scalar("Loss/Validation_CEDice", val_log['val_loss'], epoch)
            # val_dice_per_class
            for i, dice in enumerate(val_log['val_dice_per_class']):
                writer.add_scalar(f"Dice/Validation_Class_{i}", dice, epoch)

            epoch_progress.set_postfix({"Validation Dice": val_log['val_dice'], 
                                        "Validation CEDiceLoss": val_log['val_loss']})

            # Model Checkpointing
            if val_log['val_dice'] > best_val_dice:
                best_val_dice = val_log['val_dice']
                state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
                torch.save(state_dict, model_dir / 'best_model.pt')
                print(f"=> New best model saved with Dice: {best_val_dice:.4f}")

    writer.close()
    wandb.finish()
    print(f"\nTraining finished. Best validation Dice: {best_val_dice:.4f}")

if __name__ == '__main__':
    main()