# File: test_flare_3d.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import argparse
import os
import time
import SimpleITK as sitk
from datasets import load_dataset
# Import the necessary metrics from MONAI
from monai.metrics import DiceMetric, SurfaceDiceMetric
from torch.utils.data import DataLoader
# --- Project-specific imports ---
from unet import Unet, Unet_SimCLR
from dataset import NiftiDataset
from local_flare_loader import LocalFlareTask3

# Helper class for averaging metrics
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def parse_args():
    # ... (This function remains exactly the same as the previous version) ...
    parser = argparse.ArgumentParser(description="3D Inference for FLARE Segmentation")
    parser.add_argument('--name', required=True, help='Name of the trained model directory in ./models/')
    parser.add_argument('--test_domain', default='mri', choices=['mri', 'pet'], help='Which validation set to use for testing.')
    parser.add_argument('--simclr', action='store_true', help='Set this flag if the saved model is a Unet_SimCLR checkpoint.')
    parser.add_argument('--device', default='cuda:0', help='Device to use for inference.')
    parser.add_argument('--save_preds', action='store_true', help='Set this flag to save predicted segmentations as NIfTI files.')
    args = parser.parse_args()
    args.input_channel = 1
    if args.test_domain == 'mri': args.output_channel = 14
    elif args.test_domain == 'pet': args.output_channel = 5
    return args


def test_3d_volume(args, model, test_loader):
    """
    Performs slice-by-slice inference and computes accuracy and efficiency metrics.
    """
    model.eval()
    
    # --- 1. Initialize all metrics and meters ---
    # Accuracy Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    # SurfaceDice requires voxel spacing to calculate distances in mm.
    # We set a tolerance of 1mm, which is a common setting.
    surface_dice_metric = SurfaceDiceMetric(include_background=False, class_thresholds=[1.0], reduction="mean")

    # Efficiency Metrics
    time_meter = AverageMeter()
    gpu_mem_meter = AverageMeter()
    
    output_dir = os.path.join('./output', args.name, args.test_domain)
    if args.save_preds:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Predictions will be saved to: {output_dir}")

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            image_3d = batch['image'].to(args.device)
            label_3d = batch['label'] # Keep label on CPU for now
            
            original_itk_image = sitk.ReadImage(batch['image_path'][0])
            voxel_spacing = original_itk_image.GetSpacing()

            # --- 2. Measure Efficiency ---
            torch.cuda.reset_peak_memory_stats(args.device)
            start_time = time.time()

            # --- Inference Loop (same as before) ---
            depth = image_3d.shape[2]
            predicted_volume = torch.zeros_like(label_3d).squeeze(0)
            for z_slice in range(depth):
                image_slice = image_3d[:, :, z_slice, :, :]
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(image_slice)
                    if isinstance(output, tuple): output = output[1]
                pred_slice = torch.argmax(output, dim=1)
                predicted_volume[z_slice, :, :] = pred_slice.squeeze(0).cpu()
            
            end_time = time.time()
            inference_time = end_time - start_time
            peak_gpu_mem = torch.cuda.max_memory_allocated(args.device) / (1024 * 1024) # in MB
            
            time_meter.update(inference_time)
            gpu_mem_meter.update(peak_gpu_mem)

            # --- 3. Compute Accuracy Metrics ---
            # Move data to device and format for metrics
            predicted_volume = predicted_volume.long().to(args.device)
            label_3d = label_3d.long().to(args.device)

            pred_one_hot = F.one_hot(predicted_volume, num_classes=args.output_channel).permute(3, 0, 1, 2).unsqueeze(0)
            label_one_hot = F.one_hot(label_3d.squeeze(0), num_classes=args.output_channel).permute(3, 0, 1, 2).unsqueeze(0)

            # Update metrics
            dice_metric(y_pred=pred_one_hot, y=label_one_hot)
            surface_dice_metric(y_pred=pred_one_hot, y=label_one_hot, spacing=voxel_spacing)
            
            # --- Save Prediction (optional) ---
            if args.save_preds:
                # ... (save logic remains the same)
                pred_np = predicted_volume.cpu().numpy().astype(np.uint8)
                pred_itk = sitk.GetImageFromArray(pred_np)
                pred_itk.CopyInformation(original_itk_image)
                original_filename = os.path.basename(batch['image_path'][0])
                save_path = os.path.join(output_dir, f"pred_{original_filename}")
                sitk.WriteImage(pred_itk, save_path)

    # --- 4. Aggregate and return all metrics ---
    metrics = {
        "dice": dice_metric.aggregate().item(),
        "nsd": surface_dice_metric.aggregate().item(),
        "time_s": time_meter.avg,
        "gpu_mem_mb": gpu_mem_meter.avg,
    }
    dice_metric.reset()
    surface_dice_metric.reset()
    
    return metrics


def main():
    args = parse_args()
    # ... (Setup and Model Loading logic remains the same) ...
    print("--- Configuration ---")
    for arg in vars(args): print(f"{arg}: {getattr(args, arg)}")
    print("---------------------")

    model = Unet(in_channel=args.input_channel, out_channel=args.output_channel).to(args.device)
    checkpoint_path = os.path.join('./models', args.name, 'best_val_dice_model.pt')
    if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading model from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=args.device)
    if args.simclr:
        print("SimCLR flag set. Stripping projector layers...")
        model_keys = model.state_dict().keys()
        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(state_dict)

    LOCAL_DATASET_PATH = "/path/to/your/data/FLARE-Task3-DomainAdaption"
    config_name = f"validation_{args.test_domain}"
    print(f"Loading test data: {config_name}")
    hf_test_dataset = load_dataset("./local_flare_loader.py", name=config_name, data_dir=LOCAL_DATASET_PATH, trust_remote_code=True)["train"]
    test_dataset = NiftiDataset(hf_test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # --- Run Inference and get the results dictionary ---
    results = test_3d_volume(args, model, test_loader)
    
    # --- 5. Print the formatted results ---
    print("\n--- 3D Inference Complete ---")
    print(f"  Model: {args.name}")
    print(f"  Test Domain: {args.test_domain}")
    print("\n  Accuracy Metrics:")
    print(f"    - Mean Foreground Dice (DSC):      {results['dice']:.4f}")
    print(f"    - Mean Surface Dice (NSD-like):    {results['nsd']:.4f}")
    print("\n  Efficiency Metrics (per volume):")
    print(f"    - Average Inference Time:          {results['time_s']:.2f} seconds")
    print(f"    - Average Peak GPU Memory:         {results['gpu_mem_mb']:.2f} MB")
    print("-----------------------------")


if __name__ == "__main__":
    main()