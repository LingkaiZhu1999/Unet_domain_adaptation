import torch
from unet import Unet, Unet_SimCLR
from tqdm import tqdm
import numpy as np
import argparse
from dataset import BratsTestDataset
import SimpleITK
import sys
sys.path.insert(1, '..')
from metrics import Dice
import os
from glob import glob
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='LGG_adapt_HGG_lambda_1000_batchsize_8_only_source_domain_comb_Cch_seed_1_nodropout', help='model name: (default: arch+timestamp')
    parser.add_argument('--domain', default='HGG')
    parser.add_argument('--simclr', default=True)
    parser.add_argument('--input_channel', default=4)
    parser.add_argument('--output_channel', default=3)
    parser.add_argument('--seed', default=1) #remember to change this
    args = parser.parse_args()
    return args


args = parse_args()
device = "cuda:0"
if not os.path.exists(f'../output/{args.name}'):
    os.mkdir(f'../output/{args.name}')
if not os.path.exists(f'../output/{args.name}/{args.domain}'):
    os.mkdir(f'../output/{args.name}/{args.domain}')
torch.cuda.manual_seed_all(args.seed)
model = Unet(in_channel=args.input_channel, out_channel=args.output_channel).to(device)
state_dict = torch.load(f'../models/{args.name}/best_val_dice_model.pt') 
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
    print(args.name)
    print(test_data_path)
    test_data_paths = glob(test_data_path)
    test_dataset = BratsTestDataset(datapaths=test_data_paths, augmentation=None)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=2,pin_memory=True,drop_last=False)
    diceMetric = Dice(n_class=3).to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
            
            flair = images['flair'].to(device)
            t1ce = images['t1ce'].to(device)
            t1 = images['t1'].to(device)
            t2 = images['t2'].to(device)
            predicted = torch.empty((3, 240, 240, 155)).to(device)
            _, _, _, nums_z = images['flair'].shape
            for slice_num_z in range(0, nums_z):
                flair_slice = flair[:, :, :, slice_num_z]
                t1ce_slice = t1ce[:, :, :, slice_num_z]
                t1_slice = t1[:, :, :, slice_num_z]
                t2_slice = t2[:, :, :, slice_num_z]
                images_slice = torch.concat((flair_slice, t1_slice, t1ce_slice, t2_slice), dim=0).unsqueeze(0)
                output = model(images_slice)
                # plt.imshow(output[0,:, :, :].cpu().numpy(), vmin=)
                if torch.max(output) != 0:
                    pred_post = to_lbl(output[0, :, :, :].cpu().numpy())
                    plt.imshow(pred_post, vmin=0, vmax=3)
                    plt.savefig(f"../output/{args.name}/{args.domain}/{batch_idx}_{slice_num_z}_pred_baseline.png")
                    plt.close()
                    label_post = to_lbl(label[0, :, :, :, slice_num_z].numpy())
                    plt.imshow(label_post, vmin=0, vmax=3)
                    plt.savefig(f"../output/{args.name}/{args.domain}/{batch_idx}_{slice_num_z}_label.png")
                    plt.close()
                predicted[:, :, :, slice_num_z] = output.detach()

            diceMetric.update(predicted.unsqueeze(0), label.to(device), torch.tensor(0).to(device), torch.tensor(0).to(device))
            # out = SimpleITK.GetImageFromArray(predicted)
            # SimpleITK.WriteImage(out, f'./output/{args.name}/{batch_idx}.nii')
    dice_avg, _ , _ = diceMetric.compute()
    dice_wl, dice_tc, dice_et = dice_avg
    dice_avg = torch.mean(dice_avg).detach().cpu().numpy()
    print(f"Model: {args.name} Domain: {args.domain} Average Dice: {dice_avg} Whole Tumor: {dice_wl} Tumor Core: {dice_tc} Enhanced Tumor: {dice_et}")

def to_lbl(pred):
    enh = pred[2]
    c1, c2, c3 = pred[0] > 0.5, pred[1] > 0.5, pred[2] > 0.5
    pred = (c1 > 0).astype(np.uint8)
    pred[(c2 == False) * (c1 == True)] = 2
    pred[(c3 == True) * (c1 == True)] = 3

    components, n = label(pred == 3)
    for et_idx in range(1, n + 1):
        _, counts = np.unique(pred[components == et_idx], return_counts=True)
        if 1 < counts[0] and counts[0] < 8 and np.mean(enh[components == et_idx]) < 0.9:
            pred[components == et_idx] = 1

    et = pred == 3
    if 0 < et.sum() and et.sum() < 73 and np.mean(enh[et]) < 0.9:
        pred[et] = 1
    # pred = np.transpose(pred, (2, 1, 0)).astype(np.uint8)
    return pred

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

