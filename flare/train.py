# File: train.py
import argparse
from collections import OrderedDict
from pathlib import Path

import albumentations as A
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from monai.metrics import DiceMetric
from monai.transforms import Compose, AsDiscrete
# from monai.data import CacheDataset, DataLoader
from monai.data import DataLoader, CacheDataset, PersistentDataset
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
import torch.multiprocessing as mp # Import the multiprocessing library

# --- Project-specific imports ---
from datasets import load_dataset, concatenate_datasets
from loss import MultiClassDiceCELoss
from metrics import AverageMeter  # Assuming you have this helper class

# from monai.networks.nets import Unet as Unet3D  # Use MONAI's Unet for 3D segmentation
from utils import save_validation_results  # Assuming you have this utility function


# --- 1. Configuration and Argument Parsing ---

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Training for FLARE Segmentation")
    # --- Data and Model Configuration ---
    parser.add_argument('--name', default='', help='Model name, auto-generated if left empty')
    parser.add_argument('--data_dir', type=Path, required=True, help='Path to the root FLARE dataset directory.')
    parser.add_argument('--train_domain', default="train_ct_gt", help='Training set name (e.g., "train_ct_gt").')
    parser.add_argument('--val_domain', default="validation_mri", help='Validation set name.')
    parser.add_argument('--two_d', action='store_true', help='Use 2D model instead of 3D.')
    parser.add_argument('--three_d', action='store_true', help='Use 3D model instead of 2D.')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', default=100, type=int, help='Number of total epochs to run.')
    parser.add_argument('--early_stop', default=None, type=int, help='Early stopping patience.')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Mini-batch size.')
    parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate.')
    parser.add_argument('--validate_every', default=2, type=int, help='Validation frequency (in epochs).')
    
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

def validate_epoch(model, loader, criterion, dice_metric, device, post_trans): # Add criterion and post_trans
    model.eval()
    dice_metric.reset()
    loss_meter = AverageMeter() # Let's also track validation loss

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            with autocast(device_type=device.type, dtype=torch.float16):
                # Get the raw logits from the model
                outputs = model(images)
                # Calculate validation loss
                loss = criterion(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))

            # --- MONAI's streamlined post-processing ---
            # 1. Decollate the batch into a list of channel-first tensors
            #    This is required for the transforms to work on each item individually.
            outputs_list = list(outputs)
            labels_list = list(labels)

            # 2. Apply the post-processing transforms (argmax + one-hot)
            processed_outputs = [post_trans(output_item) for output_item in outputs_list]

            # 3. The metric can now directly consume the processed outputs and labels
            dice_metric(y_pred=processed_outputs, y=labels_list)
    pred_masks = torch.stack(processed_outputs)

    # Save the visual results (pass epoch into the function)
    save_validation_results(
        images, labels, pred_masks, 1,
        output_dir=Path('./validation_images')
    )

    # Aggregate the final metrics
    val_dice_dict = dice_metric.aggregate() # aggregate() now returns a dict per class
    # To get the mean dice across all (non-background) classes:
    val_dice = val_dice_dict.mean().item()
    
    # Return a more informative dictionary
    return OrderedDict([
        ('val_loss', loss_meter.avg),
        ('val_dice', val_dice),
        ('val_dice_per_class', val_dice_dict.cpu().numpy()) # For detailed logging
    ])

# --- 3. Main Training Function ---

def main():
    wandb.login(key="3aa7107481fda070c948a0e50409228c7c142d0f")  # Replace with your actual API key
    wandb.init(project="flare_segmentation", entity="zhulingkai", sync_tensorboard=True)
    args = parse_args()
    device = torch.device(args.device)
    if args.two_d and args.three_d:
        raise ValueError("Cannot use both --two_d and --three_d flags at the same time. Please choose one.")
    if not args.two_d and not args.three_d:
        args.three_d = True
    if args.two_d:
        print("Using 2D model.")
        from dataset import OnTheFly2DDataset as FlarePatchDataset
        from unet import UNet as UNet # Ensure you have the correct import for your 2D model
        PATCH_SIZE_TRAIN = (480, 480)
        PATCH_SIZE_VAL = (240, 240)  
    else:
        print("Using 3D model.")
        from dataset import Flare3DPatchDataset as FlarePatchDataset
        from unet_3d import UNet3D as UNet # Ensure you have the correct import for your 3D model
        PATCH_SIZE_TRAIN = (224, 192, 40)
        PATCH_SIZE_VAL = (60, 60, 20)  

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

    # --- Data Loading and Augmentations -
    
    print(f"Loading training data: {args.train_domain}")
    hf_train = load_dataset("./local_flare_loader.py", name=args.train_domain, data_dir=str(args.data_dir), trust_remote_code=True)["train"]
    # train_dataset = FlarePatchDataset(hf_train, patch_size=PATCH_SIZE_TRAIN, is_train=True, is_contrastive=False)
    
    hf_train_pseudo = load_dataset("./local_flare_loader.py", name="train_ct_pseudo", data_dir=str(args.data_dir), trust_remote_code=True)["train"]
    # train_dataset_pseudo = FlarePatchDataset(hf_train_pseudo, patch_size=PATCH_SIZE_TRAIN, is_train=True, is_contrastive=False)
    # Combine the datasets
    
    print(f"Loading validation data: {args.val_domain}")
    hf_val = load_dataset("./local_flare_loader.py", name=args.val_domain, data_dir=str(args.data_dir), trust_remote_code=True)["train"]
    val_dataset = FlarePatchDataset(hf_val, patch_size=PATCH_SIZE_VAL, is_train=False, is_contrastive=False)

    # train_dataset_cached = CacheDataset(data=train_dataset, cache_rate=1.0, num_workers=args.num_workers)
    # val_dataset_cached = CacheDataset(data=val_dataset, cache_rate=1.0, num_workers=args.num_workers)
    concat_datasets = concatenate_datasets([hf_train, hf_train_pseudo])
    # concat_datasets = hf_train
    train_dataset_concat = FlarePatchDataset(concat_datasets, patch_size=PATCH_SIZE_TRAIN, is_train=True, is_contrastive=False)
    train_dataset_concate_cached = CacheDataset(data=train_dataset_concat, cache_rate=1., num_workers=args.num_workers, runtime_cache="processes")
    val_dataset_cached = CacheDataset(data=val_dataset, cache_rate=1., num_workers=args.num_workers)

    # train_dataset_concat = ConcatDataset([train_dataset_cached, train_dataset_pseudo_cached])
    train_loader_concat = DataLoader(train_dataset_concate_cached, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset_cached, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    
    # --- Model, Loss, Optimizer, and Metrics ---
    model = UNet(in_channels=1, out_channels=args.output_channel).to(device)


    print("Compiling training and validation steps...")
    try:
        torch._dynamo.config.cache_size_limit = 64
        model = torch.compile(model)
        print("Step functions compiled successfully.")
    except Exception as e:
        print(f"Compilation failed, continuing without it. Error: {e}")
    train_epoch_compiled = train_epoch
    validate_epoch_compiled = validate_epoch

    class_weights = torch.tensor([0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
    # class_weights = None

    # Pass the weights to your loss function
    criterion = MultiClassDiceCELoss(num_classes=args.output_channel, weight=class_weights)
    
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, fused=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # lr = init_lr * ( 1 - epoch / args.epochs )^ 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9)
    scaler = GradScaler()

    # --- CRITICAL FIX: Use the correct DiceMetric for validation ---
    # dice_loss = DiceLoss(include_background=False, softmax=True, reduction='mean')
    
    # dice_metric = LossMetric(loss_fn=dice_loss)
    post_trans = Compose([
        AsDiscrete(argmax=True, to_onehot=args.output_channel)
    ])

    # Define your metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
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
        
        train_log = train_epoch_compiled(model, train_loader_concat, optimizer, criterion, scaler, device)
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            val_log = validate_epoch_compiled(model, val_loader, criterion, dice_metric, device, post_trans)

            epoch_progress.set_postfix(loss=train_log['loss'], val_dice=val_log['val_dice'], val_loss=val_log['val_loss'])
            writer.add_scalar("Dice/val", val_log['val_dice'], epoch)
            writer.add_scalar("Loss/val", val_log['val_loss'], epoch)
            # add 13 classes dice scores
            for i, dice_score in enumerate(val_log['val_dice_per_class']):
                writer.add_scalar(f"Dice/val_class_{i}", dice_score, epoch)
            writer.add_scalar("Meta/lr", optimizer.param_groups[0]['lr'], epoch)
        else:
            epoch_progress.set_postfix(loss=train_log['loss'])
        
        scheduler.step()

        # Logging
        writer.add_scalar("Loss/train", train_log['loss'], epoch)
        
        
        with open(log_path, 'a') as f:
            log_data = [epoch, optimizer.param_groups[0]['lr'], train_log['loss']]
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
    wandb.finish()

if __name__ == '__main__':
    # try:
    #     mp.set_start_method('spawn', force=True)
    #     print("Multiprocessing start method set to 'spawn'.")
    # except RuntimeError:
    #     print("Multiprocessing start method already set.")
    #     pass

    # Now, call your main function
    main()