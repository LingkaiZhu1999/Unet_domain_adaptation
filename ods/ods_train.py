import os
import argparse
from glob import glob
from collections import OrderedDict
from datetime import datetime

import numpy as np
from tqdm import tqdm

import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from ods_sup_train_dataloader import ImageDataset, BaseDataset

from loss.BCEDiceLoss import BCEDiceLoss
from utils import count_params
import pandas as pd

# from torchmetrics import Dice
from metrics import Dice
import albumentations as A
from ods_unet import U_Net
from utils import draw_training_supervise
from torch.nn import init
from sklearn.model_selection import train_test_split
import cv2

'''
Supervised Training and Validating on the source domain
'''
device = torch.device('cuda:0')
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='Eye', help='model name: (default: arch+timestamp')
    parser.add_argument('--dataset', default="rimone",
                        help='dataset name')
    parser.add_argument('--mode', default='train', type=str, help='train/val/test')
    parser.add_argument('--input_channel', default=3, type=int, help='input channels')
    parser.add_argument('--output_channel', default=1, type=int, help='input channels')
    parser.add_argument('--loss', default='CrossEntropy')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=None, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--decrease_lr', default=False)
    parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False,
                        help='nesterov')
    parser.add_argument('--evaluate_frequency', default=10, type=int)
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    # add warm up
    parser.add_argument('--seed', default=1)
    args = parser.parse_args()
    args.name = f'Eye_supervise_{args.dataset}_batchsize_{args.batch_size}_seed_{args.seed}'

    return args

def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    diceMetric = Dice(n_class=1).cpu()
    model.train()
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if args.use_cuda:
            input = input.to(device)
            target = target.to(device)

        output = model(input)
        loss = criterion(output.cpu(), target.cpu().float())
        diceMetric.update(output.cpu(), target.cpu(), loss.cpu(), torch.tensor(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    dice_avg, loss_avg, _ = diceMetric.compute()
    dice_avg = torch.mean(dice_avg)
    log = OrderedDict([
        ('loss', loss_avg.detach().numpy()),
        ('dice', dice_avg.detach().numpy())
    ])
    return log

def validate(args, val_loader, model):
    diceMetric = Dice(n_class=1).cpu()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if args.use_cuda:
                input = input.to(device)
                # target = target.to(device)
            with torch.no_grad():
                output = model(input)
            diceMetric.update(output.cpu(), target.cpu(), torch.tensor(0), torch.tensor(0))
    
    dice_avg, _, _ = diceMetric.compute()
    dice_avg = torch.mean(dice_avg)
    log = OrderedDict([
        ('dice', dice_avg.detach().numpy())
    ])

    return log

def main():
    args = parse_args()
    torch.cuda.manual_seed_all(args.seed)
    #args.dataset = "datasets"
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    criterion = torch.nn.BCEWithLogitsLoss()

    cudnn.benchmark = True
    # Data loading code
    if args.dataset != 'refuge':
        train_transform = A.Compose([
        A.Resize(576, 576, always_apply=True, interpolation=cv2.INTER_AREA)],
        )
    else:
        train_transform = None
    trainDataset = ImageDataset(dataset=args.dataset, image_path=f'../{args.dataset}/crop/images/', mask_path=f'../{args.dataset}/crop/masks/', mode='train', split_path=f'../{args.dataset}/{args.dataset}_split.csv', augmentation=train_transform)
    valDataset = ImageDataset(dataset=args.dataset, image_path=f'../{args.dataset}/crop/images/', mask_path=f'../{args.dataset}/crop/masks/', mode='val', split_path=f'../{args.dataset}/{args.dataset}_split.csv', test=True, augmentation=train_transform)
    # create model
    model = U_Net(in_channels=args.input_channel, classes=args.output_channel)

    if args.use_cuda:
        model = model.to(device)

    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_loader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        valDataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'dice', 'val_loss', 'val_dice'
    ])

    best_dice = 0
    trigger = 0
    best_train_loss = 10000
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, val_loader, model)
    
        print('loss: %.4f - dice: %.4f - val_dice: %.4f'
            %(train_log['loss'], train_log['dice'], val_log['dice']))
        tmp = pd.DataFrame([[epoch,
            args.lr,
            train_log['loss'],
            train_log['dice'],
            val_log['dice'],
        ]], columns=['epoch', 'lr', 'loss', 'dice', 'val_dice'])

        # log = log.append(tmp, ignore_index=True)
        log = pd.concat([log, tmp])
        log.to_csv('models/%s/log.csv' %args.name, index=False)
        trigger += 1
        draw_training_supervise(epoch + 1, loss_sup=log['loss'].to_numpy(), dice_train=log['dice'].to_numpy(), dice_val=log['val_dice'].to_numpy(), fig_name=args.name)
        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'models/%s/best_val_dice_model.pth' %args.name)
            best_dice = val_log['dice']
            print(f"=> saved best val dice model")
            trigger = 0
        if train_log['loss'] < best_train_loss:
            torch.save(model.state_dict(), 'models/%s/best_train_loss_model.pth' %args.name)
            print("=> save the best train loss model")

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()



if __name__ == '__main__':
    main()
    



