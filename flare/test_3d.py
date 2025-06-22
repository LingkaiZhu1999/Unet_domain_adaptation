# File: test.py
import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
import SimpleITK as sitk
import torch
from monai.data import decollate_batch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, SurfaceDiceMetric
from monai.transforms import AsDiscrete
from monai.networks import one_hot
from tqdm import tqdm

# --- Project-specific imports ---
# Ensure these files are accessible from your script's location
from dataset import NiftiDataset
from datasets import load_dataset
from unet import Unet  # Assuming your refactored Unet is in unet.py


# --- Helper Classes & Functions ---

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0.0, 0.0, 0.0, 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="3D Inference for FLARE Segmentation")
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the trained model directory in ./models/')
    parser.add_argument('--data_dir', type=Path, default=Path('/scratch/work/zhul2/data/FLARE-MedFM/FLARE-Task3-DomainAdaption'),
                        help='Path to the root FLARE dataset directory (e.g., /path/to/FLARE-Task3).')
    parser.add_argument('--test_domain', type=str, default='mri', choices=['mri', 'pet'],
                        help='Which validation set to use for testing.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--save_preds', action='store_true',
                        help='Set this flag to save predicted segmentations as NIfTI files.')
    
    args = parser.parse_args()
    args.input_channel = 1
    args.output_channel = 14 if args.test_domain == 'mri' else 5
    return args

def setup_model(args: argparse.Namespace) -> torch.nn.Module:
    """Initializes and loads the pre-trained model."""
    model = Unet(in_channels=args.input_channel, out_channels=args.output_channel).to(args.device)
    
    checkpoint_path = Path('./models') / args.name / 'best_model.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    print(f"Loading model from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(state_dict)
    
    # Optional: Compile model for a potential speed-up (requires PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        
    return model

def save_prediction(pred_mask: torch.Tensor, original_image_path: str, output_dir: Path):
    """Saves a predicted volume as a NIfTI file, copying metadata from the original."""
    original_itk_image = sitk.ReadImage(original_image_path)
    
    pred_np = pred_mask.cpu().numpy().astype(np.uint8)
    
    pred_itk = sitk.GetImageFromArray(pred_np)
    pred_itk.CopyInformation(original_itk_image)
    
    original_filename = Path(original_image_path).name
    save_path = output_dir / f"pred_{original_filename}"
    sitk.WriteImage(pred_itk, str(save_path))

# --- Core Inference and Evaluation Logic ---

def test_3d_volume(args: argparse.Namespace, model: torch.nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    """Performs slice-by-slice inference and computes accuracy and efficiency metrics."""
    model.eval()

    # 1. Correctly initialize metrics. num_classes should be the total number of classes (0-13 -> 14).
    # This is important for one_hot conversion.
    dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=args.output_channel)
    # The class_thresholds for surface dice should match the number of foreground classes.
    num_foreground_classes = args.output_channel - 1
    surface_dice_metric = SurfaceDiceMetric(include_background=False, reduction="mean", class_thresholds=[0.5] * num_foreground_classes)

    time_meter, gpu_mem_meter = AverageMeter(), AverageMeter()
    
    output_dir = Path('./output') / args.name / args.test_domain
    if args.save_preds:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Predictions will be saved to: {output_dir}")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing on {args.test_domain} domain"):
            # Assuming loader gives shape (B, D, H, W) for label and (B, 1, D, H, W) for image
            image_3d = batch['image'].to(args.device) # Shape: (1, 1, D, H, W)
            label_3d = batch['label'].to(args.device) # Shape: (1, D, H, W)

            # --- Efficiency measurement setup ---
            if 'cuda' in args.device:
                torch.cuda.reset_peak_memory_stats(args.device)
            start_time = time.perf_counter()

            # --- Efficient Inference Loop ---
            _, _, D, H, W = image_3d.shape
            # Initialize a tensor for class indices, not one-hot predictions yet
            pred_volume = torch.zeros((1, D, H, W), dtype=torch.int64, device=args.device)
            for z_slice in range(D):
                image_slice = image_3d[..., z_slice, :, :] # Shape: (1, 1, H, W)
                with torch.autocast(device_type=args.device.split(':')[0], dtype=torch.float16):
                    slice_logits = model(image_slice) # Shape: (1, 14, H, W)
                
                # Argmax gives the class index for each pixel. Shape: (1, H, W)
                slice_output = torch.argmax(slice_logits, dim=1) 
                pred_volume[0, z_slice, :, :] = slice_output # Assign to the correct slice

            inference_time = time.perf_counter() - start_time
            peak_gpu_mem = torch.cuda.max_memory_allocated(args.device) / (1024**2) if 'cuda' in args.device else 0.0
            time_meter.update(inference_time)
            gpu_mem_meter.update(peak_gpu_mem)

            # --- 4. Compute Accuracy Metrics Correctly ---

            # First, add a channel dimension to both prediction and label maps.
            # Shape becomes (B, C, D, H, W) where C=1.
            pred_volume_ch = pred_volume.unsqueeze(1) # Shape: (1, 1, D, H, W)
            label_3d_ch = label_3d.unsqueeze(1)       # Shape: (1, 1, D, H, W)

            # For DiceMetric, convert both to one-hot format
            pred_one_hot = one_hot(pred_volume_ch, num_classes=args.output_channel) # Shape: (1, 14, D, H, W)
            label_one_hot = one_hot(label_3d_ch, num_classes=args.output_channel)   # Shape: (1, 14, D, H, W)
            
            dice_metric(y_pred=pred_one_hot, y=label_one_hot)

            # For SurfaceDiceMetric, use the class index maps directly (with channel dim)
            # Both pred and label should be (B, 1, D, H, W)
            print(label_3d_ch.shape, pred_volume_ch.shape, label_one_hot.shape, pred_one_hot.shape)
            surface_dice_metric(y_pred=pred_one_hot, y=label_one_hot)
            print("dice_metric:", dice_metric.aggregate().item(), "surface_dice_metric:", surface_dice_metric.aggregate().item())
            if args.save_preds:
                # Squeeze batch dimension for saving
                final_pred_mask = pred_volume.squeeze(0)
                save_prediction(final_pred_mask, batch['image_path'][0], output_dir)
    
    # Use try-except block in case a metric was never updated (e.g., no foreground labels found)
    try:
        dice_result = dice_metric.aggregate().item()
    except Exception:
        dice_result = 0.0
    
    try:
        nsd_result = surface_dice_metric.aggregate().item()
    except Exception:
        nsd_result = 0.0

    metrics = {
        "dice": dice_result,
        "nsd": nsd_result,
        "time_s": time_meter.avg,
        "gpu_mem_mb": gpu_mem_meter.avg,
    }
    dice_metric.reset()
    surface_dice_metric.reset()
    return metrics

# --- Main Execution Block ---

def main():
    args = parse_args()
    print("--- Configuration ---")
    for key, value in vars(args).items():
        print(f"{key:<15}: {value}")
    print("---------------------\n")
    
    model = setup_model(args)
    
    # config_name = f"validation_{args.test_domain}"
    config_name = "train_ct_gt"
    print(f"Loading test data: {config_name} from {args.data_dir}")
    hf_test_dataset = load_dataset(
        "./local_flare_loader.py", 
        name=config_name, 
        data_dir=str(args.data_dir), 
        trust_remote_code=True
    )["train"]
    
    test_dataset = NiftiDataset(hf_test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    results = test_3d_volume(args, model, test_loader)
    
    print("\n--- 3D Inference Complete ---")
    print(f"  Model        : {args.name}")
    print(f"  Test Domain  : {args.test_domain}")
    print("\n  Accuracy Metrics:")
    print(f"    - Mean Foreground Dice (DSC) : {results['dice']:.4f}")
    print(f"    - Mean Surface Dice (NSD)  : {results['nsd']:.4f}")
    print("\n  Efficiency Metrics (per volume):")
    print(f"    - Avg. Inference Time      : {results['time_s']:.2f} seconds")
    print(f"    - Avg. Peak GPU Memory     : {results['gpu_mem_mb']:.2f} MB")
    print("-----------------------------\n")

if __name__ == "__main__":
    main()