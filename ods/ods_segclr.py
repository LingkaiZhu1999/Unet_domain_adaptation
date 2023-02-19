import torch
import sys
sys.path.insert(1, '/home/lingkai/lingkai/simCLR_unet_brats/loss')
from nt_xent import NTXentLoss
import torch.nn.functional as F
# from loss.BCEDiceLoss import BCEDiceLoss
from ods_unet import SegCLR_U_Net
from tqdm import tqdm
import time
sys.path.insert(1, '/home/lingkai/lingkai/simCLR_unet_brats')
from utils import draw_training, draw_training_loss, draw_training_joint_on_source
from metrics import compute_dice, Dice, AverageLoss
from ods_dataloader import ImageDataset
from glob import glob
import albumentations as A
import os
import numpy as np
import pickle
import random 
from torch.cuda.amp.grad_scaler import GradScaler
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
import cv2

class SegCLR(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        if self.args.contrastive_mode == 'inter_domain':
            self.nt_xent_loss = NTXentLoss(self.device, 2 * self.args.batch_size, self.args.temperature, use_cosine_similarity=True, con_type=self.args.con_type)
        else:
            self.nt_xent_loss = NTXentLoss(self.device, self.args.batch_size, self.args.temperature, use_cosine_similarity=True, con_type=self.args.con_type)

    def calculate_contrast_loss(self, model, x1, x2):
        z1 = model(x1, only_encoder=True)
        z2 = model(x2, only_encoder=True)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)        
        contrast_loss = self.nt_xent_loss(z1, z2)

        return contrast_loss

    def joint_train_on_source_and_target(self):

        print(self.args.contrastive_mode)
        writer = SummaryWriter(log_dir=f'../models/{self.args.name}')
        
        train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Affine(scale=(1.0, 1.25), p=0.5),
        A.ColorJitter(), 
        A.Resize(576, 576, always_apply=True, interpolation=cv2.INTER_AREA)],
        )

        train_source_dataset = ImageDataset(dataset=self.args.domain_source, image_path=f'{self.args.path}/{self.args.domain_source}/crop/images/', mask_path=f'{self.args.path}/{self.args.domain_source}/crop/masks/', mode='train', split_path=f'{self.args.path}/{self.args.domain_source}/{self.args.domain_source}_split.csv', augmentation=train_transform, pair_gen=True)
        train_target_dataset = ImageDataset(dataset=self.args.domain_target, image_path=f'{self.args.path}/{self.args.domain_target}/crop/images/', mask_path=f'{self.args.path}/{self.args.domain_target}/crop/masks/', mode='all', split_path=f'{self.args.path}/{self.args.domain_target}/{self.args.domain_target}_split.csv', augmentation=train_transform, pair_gen=True)

        
        val_source_dataset = ImageDataset(dataset=self.args.domain_source, image_path=f'{self.args.path}/{self.args.domain_source}/crop/images/', mask_path=f'{self.args.path}/{self.args.domain_source}/crop/masks/', mode='val', split_path=f'{self.args.path}/{self.args.domain_source}/{self.args.domain_source}_split.csv', augmentation=train_transform, pair_gen=True, val_source=True)
        train_val_source_dataset = ConcatDataset([train_source_dataset, val_source_dataset])
        val_target_dataset = train_target_dataset

        train_source_loader = torch.utils.data.DataLoader(train_source_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)
        train_target_loader = torch.utils.data.DataLoader(train_target_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)

        val_source_loader = torch.utils.data.DataLoader(val_source_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)
        train_val_source_loader = torch.utils.data.DataLoader(train_val_source_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)
        val_target_loader = torch.utils.data.DataLoader(val_target_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)

        model = SegCLR_U_Net(in_channels=self.args.input_channel, classes=self.args.output_channel).to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss() # supervise loss function 

        if self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay, nesterov=self.args.nesterov)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=0, last_epoch=-1) # Decrease the learning rate
        scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=True)

        start_time = time.time()
        end_time = time.time()
        best_val_loss = 10000
        best_val_dice = 0
        best_val_supervise_loss = 10000
        best_val_contrastive_loss = 10000
        dice_avg_train = []
        val_loss_ = []
        loss_supervise_avg_ = []
        loss_contrast_avg_ = []
        epoch_ = []
        compute_metric = Dice(n_class=1).to(self.device)
        trigger = 0
        for epoch in range(self.args.epochs):
            epoch_.append(epoch)
            for i, data in tqdm(enumerate(train_source_loader, start=1), total=len(train_source_loader)):

                image_source1, image_source2, label_source1, label_source2 = data

                image_source1 = image_source1.to(self.device)
                image_source2 = image_source2.to(self.device)
                label_source1 = label_source1.to(self.device)

                image_target1, image_target2, label_target1, label_target2 = next(iter(train_target_loader))
                image_target1 = image_target1.to(self.device)
                image_target2 = image_target2.to(self.device)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    z11, predict_source1 = model(image_source1, only_encoder=False)
                    z22 = model(image_source2, only_encoder=True)
                    z1 = model(image_target1, only_encoder=True)
                    z2 = model(image_target2, only_encoder=True)



                    supervise_loss = criterion(predict_source1, label_source1.float())
                # supervise_loss2 = criterion(predict2.cpu(), label2.cpu())

                # supervise_loss = (supervise_loss1) / 2
                    # ---inter domain contrastive learning---
                    if self.args.contrastive_mode == 'inter_domain':
                        z1 = torch.concat((z1, z11), dim=0)
                        z2 = torch.concat((z2, z22), dim=0)   
                        z1 = F.normalize(z1, dim=1)
                        z2 = F.normalize(z2, dim=1)
                        contrast_loss = self.nt_xent_loss(z1, z2)

                    # memory.push(z1=z1, z2=z2)     
                    # ---within domain contrastive learning---
                    if self.args.contrastive_mode == 'within_domain':
                        z1 = F.normalize(z1, dim=1)
                        z2 = F.normalize(z2, dim=1)
                        z11 = F.normalize(z11, dim=1)
                        z22 = F.normalize(z22, dim=1)
                        contrast_loss_1 = self.nt_xent_loss(z1, z2) 
                        contrast_loss_2 = self.nt_xent_loss(z11, z22) 
                        contrast_loss = 0.5 * (contrast_loss_1 + contrast_loss_2)

                    # ---only on the target domain---
                    if self.args.contrastive_mode == 'only_target_domain':
                        z1 = F.normalize(z1, dim=1)
                        z2 = F.normalize(z2, dim=1)
                        contrast_loss = self.nt_xent_loss(z1, z2)
                    
                    total_loss =  self.args.lam * supervise_loss + contrast_loss
                
                # loss_.append(total_loss)
                # metrics 
                # diceValue1 = compute_dice(predict1, label1)
                # diceValue2 = compute_dice(predict2, label2)
                # avg_dice = (diceValue1 + diceValue2) / 2 
                # backprop
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # print(torch.cuda.memory_summary(device=self.device))
                # update weights

                # optimizer.step()

                compute_metric.update(predict=predict_source1, label=label_source1, loss_sup=supervise_loss, loss_con=contrast_loss)
            trigger += 1
            dice_avg, loss_supervise_avg, loss_contrast_avg = compute_metric.compute()
            dice_avg = torch.mean(dice_avg).item()
            loss_supervise_avg = loss_supervise_avg.item()
            loss_contrast_avg = loss_contrast_avg.item()
            dice_avg_train.append(dice_avg)
            loss_supervise_avg_.append(loss_supervise_avg)
            loss_contrast_avg_.append(loss_contrast_avg)

            # tensorboard
            writer.add_scalar("Loss/train_contrastive", loss_contrast_avg, epoch)
            writer.add_scalar("Loss/train_supervise", loss_supervise_avg, epoch)
            writer.add_scalar("Loss/train_total_loss", self.args.lam * loss_supervise_avg + loss_contrast_avg, epoch)
            writer.add_scalar("Metric/train_dice", dice_avg, epoch)
            
            print('Training: Lr: {} Epoch [{:0>3} / {:0>3}] Total Loss: {:.4f} Supervise Loss: {:.4f} Contrastive Loss {:.4f} Dice: {:.4f}'.format(
                optimizer.param_groups[0]['lr'], epoch + 1, self.args.epochs, self.args.lam * loss_supervise_avg + loss_contrast_avg, loss_supervise_avg, loss_contrast_avg, dice_avg
            ))
            
            compute_metric.reset()

            if epoch % self.args.validate_frequency == 0:
                start_time = time.time()
                val_dice, val_supervise = self.validate_source_domain(val_source_loader, model, criterion, epoch, writer)
                if self.args.contrastive_mode == 'within_domain':
                    val_contrastive_source = self.validate_contrastive(model=model, val_loader1=val_source_loader, val_source=True)
                    val_contrastive_target = self.validate_contrastive(model=model, val_loader1=val_target_loader)
                    val_contrastive = 0.5 * (val_contrastive_source + val_contrastive_target) # separately compute source and target domain con loss
                if self.args.contrastive_mode == 'inter_domain':
                    val_contrastive = self.validate_contrastive(model=model, val_loader1=val_source_loader, val_loader2=val_target_loader) # together source and target domains 
                    # val_contrastive_target = self.validate_contrastive(model=model, val_loader1=val_target_loader)
                if self.args.contrastive_mode == 'only_target_domain':
                    val_contrastive = self.validate_contrastive(model=model, val_loader1=val_target_loader) 
                    # val_contrastive_target = self.validate_contrastive(model=model, val_loader1=val_target_loader) # only target domain contrastive validation
                val_loss = val_contrastive + self.args.lam * val_supervise

                val_loss_.append(val_loss.cpu().detach().numpy())
                writer.add_scalar("Loss/val_supervise", val_supervise, epoch)
                writer.add_scalar("Loss/val_contrastive", val_contrastive, epoch)
                writer.add_scalar("Loss/val_total", val_loss, epoch)
                writer.add_scalar("Metric/val_dice", val_dice, epoch)
                df = pd.DataFrame({
                    'epoch': epoch_,
                    'training_dice': dice_avg_train,
                    'supervise_loss': loss_supervise_avg_,
                    'contrastive_loss': loss_contrast_avg_,
                    'total_training_loss': np.add(np.array(loss_contrast_avg_), self.args.lam * np.array(loss_supervise_avg_)),
                    'total_val_loss': val_loss_
                })
                df.to_csv(f'../models/{self.args.name}/log.csv', index=False)
                end_time = time.time()
                draw_training(epoch + 1, loss_supervise_avg_, loss_contrast_avg_, dice_avg_train, val_loss_, fig_name=self.args.name, lam=self.args.lam)
                if val_loss < best_val_loss:
                    trigger = 0
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(f'../models/{self.args.name}', 'best_total_val_loss_model.pt'))
                    print("Saving best model")
                if val_supervise < best_val_supervise_loss:
                    best_val_supervise_loss = val_supervise
                    torch.save(model.state_dict(), os.path.join(f'../models/{self.args.name}', 'best_supervise_val_loss_model.pt'))
                # if val_contrastive_target < best_val_contrastive_loss:
                #     best_val_contrastive_loss = val_contrastive
                #     torch.save(model.state_dict(), os.path.join(f'../models/{self.args.name}', 'best_contrastive_val_loss_model.pt'))
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    torch.save(model.state_dict(), os.path.join(f'../models/{self.args.name}', 'best_val_dice_model.pt'))
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Loss: {:.4f} Time: {:.2f}s \n".format(
                    epoch + 1, self.args.epochs, val_loss,
                    (end_time - start_time)))
            # early stopping
            if self.args.early_stop is not None:
                if trigger >= self.args.early_stop:
                    print("=> early stopping")
                    break
            if epoch >= self.args.warm_up and self.args.lr_annealing:
                scheduler.step()
            writer.flush()
        writer.close()
    
    def joint_train_on_source(self):
        print(self.args.contrastive_mode)
        writer = SummaryWriter(log_dir=f'../models/{self.args.name}')
        print('Only Source Domain')
        train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Affine(scale=(1.0, 1.25), p=0.5),
        A.ColorJitter(), 
        A.Resize(576, 576, always_apply=True, interpolation=cv2.INTER_AREA)],
        )

        train_source_dataset = ImageDataset(dataset=self.args.domain_source, image_path=f'{self.args.path}/{self.args.domain_source}/crop/images/', mask_path=f'{self.args.path}/{self.args.domain_source}/crop/masks/', mode='train', split_path=f'{self.args.path}/{self.args.domain_source}/{self.args.domain_source}_split.csv', augmentation=train_transform, pair_gen=True)
       

        
        val_source_dataset = ImageDataset(dataset=self.args.domain_source, image_path=f'{self.args.path}/{self.args.domain_source}/crop/images/', mask_path=f'{self.args.path}/{self.args.domain_source}/crop/masks/', mode='val', split_path=f'{self.args.path}/{self.args.domain_source}/{self.args.domain_source}_split.csv', augmentation=train_transform, pair_gen=True, val_source=True)
        train_val_source_dataset = ConcatDataset([train_source_dataset, val_source_dataset])

        train_source_loader = torch.utils.data.DataLoader(train_source_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)

        val_source_loader = torch.utils.data.DataLoader(val_source_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)
        train_val_source_loader = torch.utils.data.DataLoader(train_val_source_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)

        model = SegCLR_U_Net(in_channels=self.args.input_channel, classes=self.args.output_channel).to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss() # supervise loss function 

        if self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay, nesterov=self.args.nesterov)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=0, last_epoch=-1) # Decrease the learning rate
        scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=True)

        start_time = time.time()
        end_time = time.time()
        best_val_loss = 10000
        best_val_dice = 0
        best_val_supervise_loss = 10000
        best_val_contrastive_loss = 10000
        dice_avg_train = []
        val_loss_ = []
        loss_supervise_avg_ = []
        loss_contrast_avg_ = []
        epoch_ = []
        compute_metric = Dice(n_class=1).to(self.device)
        trigger = 0
        for epoch in range(self.args.epochs):
            epoch_.append(epoch)
            for i, data in tqdm(enumerate(train_source_loader, start=1), total=len(train_source_loader)):

                image_source1, image_source2, label_source1, label_source2 = data

                image_source1 = image_source1.to(self.device)
                image_source2 = image_source2.to(self.device)
                label_source1 = label_source1.to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    z1, predict_source1 = model(image_source1, only_encoder=False)
                    z2 = model(image_source2, only_encoder=True)
                    z1 = F.normalize(z1, dim=1)
                    z2 = F.normalize(z2, dim=1)

                    supervise_loss = criterion(predict_source1, label_source1.float())

                    contrast_loss = self.nt_xent_loss(z1, z2) 

                    # ---only on the target domain---
                    total_loss =  self.args.lam * supervise_loss + contrast_loss
                
                # backprop
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # print(torch.cuda.memory_summary(device=self.device))
                # update weights

                # optimizer.step()

                compute_metric.update(predict=predict_source1, label=label_source1, loss_sup=supervise_loss, loss_con=contrast_loss)
            trigger += 1
            dice_avg, loss_supervise_avg, loss_contrast_avg = compute_metric.compute()
            dice_avg = torch.mean(dice_avg).item()
            loss_supervise_avg = loss_supervise_avg.item()
            loss_contrast_avg = loss_contrast_avg.item()
            dice_avg_train.append(dice_avg)
            loss_supervise_avg_.append(loss_supervise_avg)
            loss_contrast_avg_.append(loss_contrast_avg)

            # tensorboard
            writer.add_scalar("Loss/train_contrastive", loss_contrast_avg, epoch)
            writer.add_scalar("Loss/train_supervise", loss_supervise_avg, epoch)
            writer.add_scalar("Loss/train_total_loss", self.args.lam * loss_supervise_avg + loss_contrast_avg, epoch)
            writer.add_scalar("Metric/train_dice", dice_avg, epoch)
            
            print('Training: Lr: {} Epoch [{:0>3} / {:0>3}] Total Loss: {:.4f} Supervise Loss: {:.4f} Contrastive Loss {:.4f} Dice: {:.4f}'.format(
                optimizer.param_groups[0]['lr'], epoch + 1, self.args.epochs, self.args.lam * loss_supervise_avg + loss_contrast_avg, loss_supervise_avg, loss_contrast_avg, dice_avg
            ))
            
            compute_metric.reset()

            if epoch % self.args.validate_frequency == 0:
                start_time = time.time()
                val_dice, val_supervise = self.validate_source_domain(val_source_loader, model, criterion, epoch, writer)
                val_contrastive_source = self.validate_contrastive(model=model, val_loader1=val_source_loader, val_source=True)
                val_loss = val_contrastive_source + self.args.lam * val_supervise

                val_loss_.append(val_loss.cpu().detach().numpy())
                writer.add_scalar("Loss/val_supervise", val_supervise, epoch)
                writer.add_scalar("Loss/val_contrastive", val_contrastive_source, epoch)
                writer.add_scalar("Loss/val_total", val_loss, epoch)
                writer.add_scalar("Metric/val_dice", val_dice, epoch)
                df = pd.DataFrame({
                    'epoch': epoch_,
                    'training_dice': dice_avg_train,
                    'supervise_loss': loss_supervise_avg_,
                    'contrastive_loss': loss_contrast_avg_,
                    'total_training_loss': np.add(np.array(loss_contrast_avg_), self.args.lam * np.array(loss_supervise_avg_)),
                    'total_val_loss': val_loss_
                })
                df.to_csv(f'../models/{self.args.name}/log.csv', index=False)
                end_time = time.time()
                # draw_training(epoch + 1, loss_supervise_avg_, loss_contrast_avg_, dice_avg_train, val_loss_, fig_name=self.args.name, lam=self.args.lam)
                if val_loss < best_val_loss:
                    trigger = 0
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(f'../models/{self.args.name}', 'best_total_val_loss_model.pt'))
                    print("Saving best model")
                if val_supervise < best_val_supervise_loss:
                    best_val_supervise_loss = val_supervise
                    torch.save(model.state_dict(), os.path.join(f'../models/{self.args.name}', 'best_supervise_val_loss_model.pt'))
                # if val_contrastive_target < best_val_contrastive_loss:
                #     best_val_contrastive_loss = val_contrastive
                #     torch.save(model.state_dict(), os.path.join(f'../models/{self.args.name}', 'best_contrastive_val_loss_model.pt'))
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    torch.save(model.state_dict(), os.path.join(f'../models/{self.args.name}', 'best_val_dice_model.pt'))
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Loss: {:.4f} Time: {:.2f}s \n".format(
                    epoch + 1, self.args.epochs, val_loss,
                    (end_time - start_time)))
            # early stopping
            if self.args.early_stop is not None:
                if trigger >= self.args.early_stop:
                    print("=> early stopping")
                    break
            if epoch >= self.args.warm_up and self.args.lr_annealing:
                scheduler.step()
            writer.flush()
        writer.close()
             
    def validate_contrastive(self, model, val_loader1, val_loader2=None, val_source=False):
        model.eval()
        with torch.no_grad():
            avg_loss = 0
            counter = 0
            if val_source:
                for batch_idx, data in tqdm(enumerate(val_loader1), total=len(val_loader1)):
                    _, _, image_val1, image_val2, _, _ = data
                    image_val1 = image_val1.to(self.device)
                    image_val2 = image_val2.to(self.device)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        con_loss = self.calculate_contrast_loss(model, image_val1, image_val2)
                    avg_loss += con_loss.cpu()
                    
                    counter += 1

                avg_loss /= counter
            else:
                for batch_idx, data in tqdm(enumerate(val_loader1), total=len(val_loader1)):
                    image_val1, image_val2, _, _ = data
                    image_val1 = image_val1.to(self.device)
                    image_val2 = image_val2.to(self.device)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        con_loss = self.calculate_contrast_loss(model, image_val1, image_val2)
                    avg_loss += con_loss.cpu()
                    
                    counter += 1

                avg_loss /= counter
        model.train()
        return avg_loss

    def validate_source_domain(self, val_loader, model, criterion, epoch, writer):
        diceMetric = Dice(n_class=1).cpu()
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            for i, (input, target, _, _, _, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
                input = input.to(self.device)
                target = target.to(self.device)
                with torch.no_grad():
                    project_head, output = model(input, only_encoder=False)
                    loss = criterion(output, target.float())
                    if i == 0:
                        writer.add_image('Validate/Input', input[0], epoch)
                        writer.add_image('Validate/output', (torch.sigmoid(output[0]) > 0.5).int(), epoch)
                        writer.add_image('Validate/Mask', target[0], epoch)
                diceMetric.update(output.cpu(), target.cpu(), loss.detach().cpu(), torch.tensor(0))
        dice_avg, supervised_loss, _ = diceMetric.compute()
        model.train()
        return dice_avg.cpu().detach().numpy(), supervised_loss.cpu().detach().numpy()







