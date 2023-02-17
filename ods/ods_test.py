import os
from ods_unet import U_Net
import torch
from glob import glob
from ods_sup_train_dataloader import ImageDataset
from tqdm import tqdm
from metrics import dice_coef
import numpy as np
import pickle
from metrics import Dice
torch.set_printoptions(threshold=10000)
import argparse
import albumentations as A
import cv2

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='Eye_refuge_adapt_rimone_lambda_1000_batchsize_8_within_domain_aug_Cch_seed_1', help='model name: (default: arch+timestamp')
    parser.add_argument('--dataset', default="rimone",
                        help='dataset name')
    parser.add_argument('--mode', default='test', type=str, help='train/val/test')
    parser.add_argument('--input_channel', default=3, type=int, help='input channels')
    parser.add_argument('--output_channel', default=1, type=int, help='input channels')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fine_tuning', default=False, type=bool)
    # add warm up
    parser.add_argument('--seed', default=1)
    args = parser.parse_args()
    # args.name = f'Eye_supervise_{args.dataset}_batchsize_{args.batch_size}_seed_{args.seed}'

    return args

args = parse_args()

def main(args):
    print(args.name)
    print(args.dataset)
    model_name = 'best_val_dice_model.pt'
    # best_val_dice_model.pt
    # best_total_val_loss_model.pt
    print(model_name)
    if args.dataset != 'refuge':
        resize_transform = A.Compose([
        A.Resize(576, 576, always_apply=True, interpolation=cv2.INTER_AREA)],
        )
    else: 
        resize_transform = None
    torch.cuda.manual_seed_all(args.seed)
    model = U_Net(in_channels=args.input_channel, classes=args.output_channel).to(args.device)
    state_dict = torch.load(f'models/{args.name}/{model_name}') 
    keys = []
    for k, v in state_dict.items():
        if k.startswith('projector'):
            continue
        keys.append(k)
    state_dict = {k: state_dict[k] for k in keys}
    model.load_state_dict(state_dict)
    model = model.to(args.device)

    test_dataset =  ImageDataset(dataset=args.dataset, image_path=f'../{args.dataset}/crop/images/', mask_path=f'../{args.dataset}/crop/masks/', mode='test', split_path=f'../{args.dataset}/{args.dataset}_split.csv', test=True, augmentation=resize_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    model.eval()
    diceMetric = Dice(n_class=1).cpu()
    # switch to evaluate mode
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            input = input.to(args.device)
            target = target.to(args.device)
            output = model(input)
            diceMetric.update(output.cpu(), target.cpu(), torch.tensor(0), torch.tensor(0))
    
    dice_avg, _, _ = diceMetric.compute()
    dice_avg = torch.mean(dice_avg)
    print(dice_avg)

if __name__ == "__main__":
    main(args)




