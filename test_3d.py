import torch
from unet import Unet, Unet_SimCLR
from tqdm import tqdm
import numpy as np
import argparse
from dataset import BratsTestDataset, BratsTestDataset_one_modularity
import SimpleITK
from metrics import Dice
import os
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='Joint_HGG', help='model name: (default: arch+timestamp')
    parser.add_argument('--domain', default='HGG')
    parser.add_argument('--simclr', default=True)
    parser.add_argument('--input_channel', default=4)
    parser.add_argument('--output_channel', default=3)
    args = parser.parse_args()

    return args


args = parse_args()
device = "cuda:1"
if not os.path.exists(f'./output/{args.name}'):
    os.mkdir(f'./output/{args.name}')
torch.cuda.manual_seed_all(1)
model = Unet(in_channel=args.input_channel, out_channel=args.output_channel).to(device)
state_dict = torch.load(f'models/{args.name}/best_dice_model.pt') #350
keys = []
if args.simclr:
    for k, v in state_dict.items():
        if k.startswith('projector'):
            continue
        keys.append(k)
state_dict = {k: state_dict[k] for k in keys}
model.load_state_dict(state_dict)

def test_with_all_modularity(args, device, model):
    test_data_path = f"/mnt/asgard2/data/lingkai/braTS20/{args.domain}/Test/*"
    test_data_paths = glob(test_data_path)
    test_dataset = BratsTestDataset(datapaths=test_data_paths, augmentation=None)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,pin_memory=True,drop_last=False)
    diceMetric = Dice().cpu()
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, label) in tqdm(enumerate(test_loader), total=len(test_loader)):

            predicted = torch.empty((3, 240, 240, 155))
            _, _, _, nums_z = images['flair'].shape
            for slice_num_z in range(0, nums_z):
                flair_slice = images['flair'][:, :, :, slice_num_z]
                t1ce_slice = images['t1ce'][:, :, :, slice_num_z]
                t1_slice = images['t1'][:, :, :, slice_num_z]
                t2_slice = images['t2'][:, :, :, slice_num_z]
                images_slice = np.concatenate((flair_slice, t1_slice, t1ce_slice, t2_slice), axis=0)
                images_slice = torch.from_numpy(np.expand_dims(images_slice, 0)).to(device)
                output = model(images_slice)
                predicted[:, :, :, slice_num_z] = output.cpu().detach()
            diceMetric.update(predicted.unsqueeze(0), label, torch.tensor(0), torch.tensor(0))
            out = SimpleITK.GetImageFromArray(predicted)
            SimpleITK.WriteImage(out, f'./output/{args.name}/{batch_idx}.nii')
    dice_avg, _ , _ = diceMetric.compute()
    dice_wl, dice_tc, dice_et = dice_avg
    dice_avg = torch.mean(dice_avg).detach().numpy()
    print(f"Model: {args.name} Domain: {args.domain} Average Dice: {dice_avg} Whole Tumor: {dice_wl} Tumor Core: {dice_tc} Enhanced Tumor: {dice_et}")

def test_with_one_modularity(args, device, model):
    test_data_path = f"/mnt/asgard2/data/lingkai/braTS20/{args.domain}/Test/*"
    test_data_paths = glob(test_data_path)
    test_dataset = BratsTestDataset(datapaths=test_data_paths, augmentation=None)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,pin_memory=True,drop_last=False)
    diceMetric = Dice().cpu()
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, label) in tqdm(enumerate(test_loader), total=len(test_loader)):

            predicted = torch.empty((3, 240, 240, 155))
            _, _, _, nums_z = images['flair'].shape
            for slice_num_z in range(0, nums_z):
                image_slice = images['t2'][:, :, :, slice_num_z]
                image_slice = torch.from_numpy(np.expand_dims(image_slice, 0)).to(device)
                output = model(image_slice)
                predicted[:, :, :, slice_num_z] = output.cpu().detach()
            diceMetric.update(predicted.unsqueeze(0), label, 0, 0)
            out = SimpleITK.GetImageFromArray(predicted)
            SimpleITK.WriteImage(out, f'./output/{args.name}/{batch_idx}.nii')
    dice_avg, _ , _ = diceMetric.compute()
    dice_wl, dice_tc, dice_et = dice_avg
    dice_avg = torch.mean(dice_avg).detach().numpy()
    print(f"Model: {args.name} Domain: {args.domain} Average Dice: {dice_avg} Whole Tumor: {dice_wl} Tumor Core: {dice_tc} Enhanced Tumor: {dice_et}")


if __name__ == "__main__":
    # test_with_one_modularity(args, device, model)
    test_with_all_modularity(args, device, model)
