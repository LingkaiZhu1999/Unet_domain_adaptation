# File: run_flare.py

import argparse
import torch
import os
import torch.backends.cudnn as cudnn

# Import the main training class from your adapted training script
from segclr import SegCLR_FLARE

def parse_args():
    parser = argparse.ArgumentParser(description="Run Domain Adaptation Training for FLARE 2024")

    # --- Domain and Model Configuration ---
    parser.add_argument('--name', default='', help='model name, auto-generated if left empty')
    parser.add_argument('--source_domain', default="ct", choices=['ct'],
                        help='Source domain for training. Currently only CT is supported.')
    parser.add_argument('--target_domain', default="mri", choices=['mri', 'pet'],
                        help='Target domain for adaptation.')
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping patience (default: 20)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='optimizer type (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--warm_up', default=5, type=int,
                        help='number of warm-up epochs')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay for optimizer')
    
    # --- Loss and Contrastive Learning Parameters ---
    parser.add_argument('--lam', default=1.0, type=float,
                        help='lambda weight for the supervised loss term')
    parser.add_argument('--temperature', default=0.1, type=float,
                        help='temperature parameter for NTXentLoss')
    parser.add_argument('--contrastive_mode', default='inter_domain',
                        choices=['inter_domain', 'within_domain', 'only_target_domain'],
                        help='contrastive learning strategy')

    # --- System and Reproducibility ---
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training (e.g., cuda:0, cuda:1)')
    parser.add_argument('--validate_frequency', default=1, type=int,
                        help='run validation every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')

    args = parser.parse_args()
    
    # --- Derive model parameters and generate name automatically ---
    args.input_channel = 1  # All FLARE modalities are single-channel

    if args.target_domain == 'mri':
        args.output_channel = 14  # 13 organs + 1 background
    elif args.target_domain == 'pet':
        args.output_channel = 5   # 4 organs + 1 background
        
    if args.name == '':
        args.name = (
            f"FLARE_{args.source_domain}_to_{args.target_domain}_"
            f"lambda_{args.lam}_bs_{args.batch_size}_"
            f"{args.contrastive_mode}_seed_{args.seed}"
        )

    return args

def main():
    args = parse_args()

    # --- Setup Environment ---
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    cudnn.benchmark = True

    # --- Create Directories ---
    model_dir = os.path.join('./models', args.name)
    output_dir = os.path.join('./output', args.name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # --- Save Configuration ---
    with open(os.path.join(model_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            line = f"{arg}: {getattr(args, arg)}\n"
            f.write(line)
            print(line.strip())

    # --- Initialize and Run Training ---
    trainer = SegCLR_FLARE(args)
    print("\n--- Starting Joint Domain Adaptation Training ---")
    trainer.joint_train_on_source_and_target()
    print("--- Training Finished ---")


if __name__ == "__main__":
    main()