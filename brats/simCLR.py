import torch
from loss.nt_xent import NTXentLoss
import torch.nn.functional as F
from loss.BCEDiceLoss import BCEDiceLoss
from unet import Unet_SimCLR, Encoder_SimCLR, Unet
from tqdm import tqdm
import time
from metrics import compute_dice, Dice, AverageLoss
from dataset import BratsTrainContrastDataset, BratsSuperviseTrainDataset, BratsTestDataset, BratsTrainContrastDataset_only_data_aug
from glob import glob
import albumentations as A
import os
import numpy as np
from utils import draw_training, draw_training_loss, draw_training_joint_on_source
import pickle
import random 
from torch.cuda.amp.grad_scaler import GradScaler
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)

class SimCLR(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        if self.args.contrastive_mode == 'inter_domain':
            self.nt_xent_loss = NTXentLoss(self.device, 2 * self.args.batch_size, self.args.temperature, use_cosine_similarity=True)
        else:
            self.nt_xent_loss = NTXentLoss(self.device, self.args.batch_size, self.args.temperature, use_cosine_similarity=True)

    def calculate_contrast_loss(self, model, x1, x2):
        z1 = model(x1, only_encoder=True)
        z2 = model(x2, only_encoder=True)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)        
        contrast_loss = self.nt_xent_loss(z1, z2)

        return contrast_loss

    def joint_train_on_source_and_target(self):

        print(self.args.contrastive_mode)
        writer = SummaryWriter(log_dir=f'./models/{self.args.name}')

        train_transform = A.Compose([
        # A.RandomCrop(160, 160, always_apply=True), # decrease performance
        # A.CenterCrop(160, 160, always_apply=True),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(1.0, 1.5), p=0.5),
        A.Affine(translate_percent=(0, 0.25), p=0.5),
        # A.CropNonEmptyMaskIfExists(width=120, height=120),
        A.ColorJitter(brightness=0.6),
        # A.Resize(160, 160, always_apply=True)
        ],
        additional_targets = {'t1': 'image', 't1ce': 'image', 't2': 'image', 'tumorCore': 'mask', 'enhancingTumor': 'mask'}
        )

        train_source_paths = glob(f'/home/lingkai/lingkai/braTS20/{self.args.domain_source}/Train/*')
        val_source_paths = glob(f'/home/lingkai/lingkai/braTS20/{self.args.domain_source}/Val/*')
        # val_source_paths = np.append(train_source_paths, val_source_paths)
        train_target_paths = glob(f'/home/lingkai/lingkai/braTS20/{self.args.domain_target}/All/*')
        val_target_paths = train_target_paths

        train_source_dataset = BratsTrainContrastDataset(train_source_paths, augmentation=train_transform)
        train_target_dataset = BratsTrainContrastDataset(train_target_paths, augmentation=train_transform)

        val_target_dataset = BratsTrainContrastDataset(val_target_paths, augmentation=train_transform)
        val_source_dataset = BratsTrainContrastDataset(val_source_paths, augmentation=train_transform)
        # val_supervise_dataset = BratsSuperviseTrainDataset(val_source_paths, augmentation=train_transform)
        val_source_3d_dataset = BratsTestDataset(val_source_paths, augmentation=None)

        train_source_loader = torch.utils.data.DataLoader(train_source_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)
        train_target_loader = torch.utils.data.DataLoader(train_target_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)

        # val_supervise_loader = torch.utils.data.DataLoader(val_supervise_dataset,batch_size=self.args.batch_size,shuffle=True,pin_memory=True,drop_last=False)
        val_target_loader = torch.utils.data.DataLoader(val_target_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)
        val_source_loader = torch.utils.data.DataLoader(val_source_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)
        val_source_3d_loader = torch.utils.data.DataLoader(val_source_3d_dataset,batch_size=1,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)

        model = Unet_SimCLR(in_channel=self.args.input_channel, out_channel=self.args.output_channel).to(self.device)
        # model.apply(weights_init_kaiming) 
        criterion = BCEDiceLoss() # supervise loss function 

        optimizer = torch.optim.Adam(model.parameters(), self.args.lr, weight_decay=1e-4)
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
        compute_metric = Dice().to(self.device)
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



                    supervise_loss = criterion(predict_source1, label_source1)
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
                # compute_metric.update(predict=predict2, label=label2, loss_sup=supervise_loss2, loss_con=contrast_loss)
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
                # val_supervise = self.validate_supervise(model, val_source_loader)
                if self.args.contrastive_mode == 'within_domain':
                    val_contrastive_source = self.validate_contrastive(model=model, val_loader1=val_source_loader)
                    val_contrastive_target = self.validate_contrastive(model=model, val_loader1=val_target_loader)
                    val_contrastive = 0.5 * (val_contrastive_source + val_contrastive_target) # separately compute source and target domain con loss
                if self.args.contrastive_mode == 'inter_domain':
                    val_contrastive = self.validate_contrastive(model=model, val_loader1=val_source_loader, val_loader2=val_target_loader) # together source and target domains 
                    # val_contrastive_target = self.validate_contrastive(model=model, val_loader1=val_target_loader)
                if self.args.contrastive_mode == 'only_target_domain':
                    val_contrastive = self.validate_contrastive(model=model, val_loader1=val_target_loader) 
                    # val_contrastive_target = self.validate_contrastive(model=model, val_loader1=val_target_loader) # only target domain contrastive validation
                
                
                val_dice, val_supervised_loss = self.validate_dice(self.args, val_source_3d_loader, model, criterion)
                val_loss = (val_contrastive + self.args.lam * val_supervised_loss).detach().cpu().numpy()
                val_loss_.append(val_loss)
                writer.add_scalar("Loss/val_supervise", val_supervised_loss, epoch)
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
                df.to_csv(f'./models/{self.args.name}/log.csv', index=False)
                end_time = time.time()
                draw_training(epoch + 1, loss_supervise_avg_, loss_contrast_avg_, dice_avg_train, val_loss_, fig_name=self.args.name, lam=self.args.lam)
                if val_loss < best_val_loss:
                    trigger = 0
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(f'./models/{self.args.name}', 'best_total_val_loss_model.pt'))
                    print("Saving best model")
                if val_supervised_loss < best_val_supervise_loss:
                    best_val_supervise_loss = val_supervised_loss
                    torch.save(model.state_dict(), os.path.join(f'./models/{self.args.name}', 'best_supervise_val_loss_model.pt'))
                # if val_contrastive_target < best_val_contrastive_loss:
                #     best_val_contrastive_loss = val_contrastive
                    torch.save(model.state_dict(), os.path.join(f'./models/{self.args.name}', 'best_contrastive_val_loss_model.pt'))
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    torch.save(model.state_dict(), os.path.join(f'./models/{self.args.name}', 'best_val_dice_model.pt'))
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Loss: {:.4f} Time: {:.2f}s \n".format(
                    epoch + 1, self.args.epochs, val_loss,
                    (end_time - start_time)))
            # early stopping
            if self.args.early_stop is not None:
                if trigger >= self.args.early_stop:
                    print("=> early stopping")
                    break
            if epoch >= self.args.warm_up:
                scheduler.step()
            writer.flush()
        writer.close()
    
    def joint_train_on_source(self):
         
        writer = SummaryWriter(log_dir=f'./models/{self.args.name}')
        print(f"Source Domain: {self.args.domain_source} Model: {self.args.name}")
        
        train_transform = A.Compose([
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(1.0, 1.5), p=0.5),
        A.Affine(translate_percent=(0, 0.25), p=0.5),
        A.ColorJitter(brightness=0.6)],
        additional_targets={'t1': 'image', 't1ce': 'image', 't2': 'image', 'tumorCore': 'mask', 'enhancingTumor': 'mask'}
        )

        train_source_paths = glob(f'../braTS20/{self.args.domain_source}/Train/*')
        val_paths = glob(f'../braTS20/{self.args.domain_source}/Val/*')
        if self.args.mode == 'comb':
            train_source_dataset = BratsTrainContrastDataset(train_source_paths, augmentation=train_transform)
            val_con_dataset = BratsTrainContrastDataset(val_paths, augmentation=train_transform)
            print('comb')
        else:
            train_source_dataset = BratsTrainContrastDataset_only_data_aug(train_source_paths, augmentation=train_transform)
            val_con_dataset = BratsTrainContrastDataset_only_data_aug(val_paths, augmentation=train_transform)
            print('aug')
        train_source_loader = torch.utils.data.DataLoader(train_source_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)

        val_dataset = BratsTestDataset(val_paths, augmentation=None) # for dice metric
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False, num_workers=2, pin_memory=True,drop_last=False)
        val_con_loader = torch.utils.data.DataLoader(val_con_dataset, batch_size=self.args.batch_size, shuffle=True,num_workers=2,pin_memory=True,drop_last=False)
        
        model = Unet_SimCLR(in_channel=self.args.input_channel, out_channel=self.args.output_channel).to(self.device)
       
        # model.apply(weights_init_kaiming) 
        
        criterion = BCEDiceLoss().to(device=self.device) # supervise loss function 

        optimizer = torch.optim.Adam(model.parameters(), self.args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=0, last_epoch=-1) # Decrease the learning rate
        scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=True)
        
        start_time = time.time()
        end_time = time.time()
        best_val_loss = 10000
        best_val_dice = 0
        best_train_loss = 100000
        best_val_supervise_loss = 10000
        epoch_ = []
        dice_avg_train = []
        val_dice_ = []
        val_loss_ = []
        loss_supervise_avg_ = []
        loss_contrast_avg_ = []
        compute_metric = Dice().to(self.device)
        trigger = 0
        model.train()
        for epoch in range(self.args.epochs):
            epoch_.append(epoch)
            for i, data in tqdm(enumerate(train_source_loader, start=1), total=len(train_source_loader)):
                image1, image2, label1, label2 = data

                image1 = image1.to(self.device)
                image2 = image2.to(self.device)
                label1 = label1.to(self.device)
                label2 = label2.to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    z1, predict1 = model(image1)
                    z2, _ = model(image2)

                    supervise_loss = criterion(predict1, label1)
 
                    z1 = F.normalize(z1, dim=1)
                    z2 = F.normalize(z2, dim=1)

                # memory.push(z1=z1, z2=z2)     
                
                
                    contrast_loss = self.nt_xent_loss(z1, z2) 
                # dynamic lambda later
                    total_loss =  self.args.lam * supervise_loss + contrast_loss
                # backprop
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # print(torch.cuda.memory_summary(device=self.device))
                compute_metric.update(predict=predict1, label=label1, loss_sup=supervise_loss, loss_con=contrast_loss)

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
                optimizer.param_groups[0]['lr'], epoch + 1, self.args.epochs, loss_supervise_avg*self.args.lam + loss_contrast_avg, loss_supervise_avg, loss_contrast_avg, dice_avg
            ))
            
            compute_metric.reset()
            if epoch % self.args.validate_frequency == 0:
                start_time = time.time()
                val_dice, val_supervised_loss = self.validate_dice(self.args, val_loader, model, criterion)
                val_dice_.append(val_dice)
                end_time = time.time()
                val_supervise = self.validate_supervise(model, val_con_loader)
                val_contrastive = self.validate_contrastive(model, val_con_loader)
                val_loss = self.args.lam * val_supervise + val_contrastive
                val_loss_.append(val_loss)
                draw_training_joint_on_source(epoch + 1, loss_supervise_avg_, loss_contrast_avg_, dice_avg_train, val_dice_, val_loss_, fig_name=self.args.name, lam=self.args.lam)

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
                df.to_csv(f'./models/{self.args.name}/log.csv', index=False)
                
                # model selection

                if val_loss < best_val_loss:
                    trigger = 0
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(f'./models/{self.args.name}', 'best_total_val_loss_model.pt'))
                    print("Saving best model")
                if val_supervise < best_val_supervise_loss:
                    best_val_supervise_loss = val_supervise
                    torch.save(model.state_dict(), os.path.join(f'./models/{self.args.name}', 'best_supervise_val_loss_model.pt'))
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    torch.save(model.state_dict(), os.path.join(f'./models/{self.args.name}', 'best_val_dice_model.pt'))

                print("Valid:\t Epoch[{:0>3}/{:0>3}] Dice: {:.4f} Time: {:.2f}s \n".format(
                    epoch + 1, self.args.epochs, val_dice,
                    (end_time - start_time)))
            # early stopping
            if self.args.early_stop is not None:
                if trigger >= self.args.early_stop:
                    print("=> early stopping")
                    break
            if epoch >= self.args.warm_up:
                scheduler.step()
                
    def validate_contrastive(self, model, val_loader1, val_loader2=None):
        model.eval()
        with torch.no_grad():
            avg_loss = 0
            counter = 0
            for batch_idx, data in tqdm(enumerate(val_loader1), total=len(val_loader1)):
                image_val1, image_val2, _, _ = data
                image_val1 = image_val1.to(self.device)
                image_val2 = image_val2.to(self.device)
                if val_loader2 is not None:
                    image_target1, image_target2, _, _ = next(iter(val_loader2))
                    image_target1 = image_target1.to(self.device)
                    image_target2 = image_target2.to(self.device)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        z11 = model(image_val1, only_encoder=True)
                        z22 = model(image_val2, only_encoder=True)
                        z1 = model(image_target1, only_encoder=True)
                        z2 = model(image_target2, only_encoder=True)
                        z1 = torch.concat((z1, z11), dim=0)
                        z2 = torch.concat((z2, z22), dim=0)   
                        z1 = F.normalize(z1, dim=1)
                        z2 = F.normalize(z2, dim=1)
                        con_loss = self.nt_xent_loss(z1, z2)
                else:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        con_loss = self.calculate_contrast_loss(model, image_val1, image_val2)
                avg_loss += con_loss.cpu()
                
                counter += 1

            avg_loss /= counter
        model.train()
        return avg_loss
    
    def validate_supervise(self, model, val_loader):
        criterion = BCEDiceLoss()
        segclr_model_dict = model.state_dict()
        Unet_model = Unet(in_channel=self.args.input_channel, out_channel=self.args.output_channel).to(self.device)
        Unet_model_dict = Unet_model.state_dict()
        segclr_model_dict = {k: v for k, v in segclr_model_dict.items() if k in Unet_model_dict}
        Unet_model_dict.update(segclr_model_dict)
        Unet_model.load_state_dict(Unet_model_dict)
        Unet_model.eval()
        with torch.no_grad():
            avg_loss = 0
            counter = 0
            for batch_idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                image_val, _, label_val, _ = data
                image_val = image_val.to(self.device)
                predict = Unet_model(image_val)
                label_val = label_val.to(self.device)
                supervise_loss = criterion(predict, label_val)
                avg_loss += supervise_loss.cpu()
                
                counter += 1

            avg_loss /= counter
        return avg_loss
    
    def validate_dice(self, args, val_loader, model, criterion):
        diceMetric = Dice().to(self.device)
        segclr_model_dict = model.state_dict()
        Unet_model = Unet(in_channel=args.input_channel, out_channel=args.output_channel).to(self.device)
        Unet_model_dict = Unet_model.state_dict()
        segclr_model_dict = {k: v for k, v in segclr_model_dict.items() if k in Unet_model_dict}
        Unet_model_dict.update(segclr_model_dict)
        Unet_model.load_state_dict(Unet_model_dict)
        Unet_model.eval()
        with torch.no_grad():
            for batch_idx, (images, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
                label = label.to(self.device)
                predicted = torch.empty((3, 240, 240, 155)).to(self.device)
                _, _, _, nums_z = images['flair'].shape
                flair = images['flair'].to(self.device)
                t1ce = images['t1ce'].to(self.device)
                t1 = images['t1'].to(self.device)
                t2 = images['t2'].to(self.device)
                supervised_loss = 0
                for slice_num_z in range(0, nums_z):
                    flair_slice = flair[:, :, :, slice_num_z]
                    t1ce_slice = t1ce[:, :, :, slice_num_z]
                    t1_slice = t1[:, :, :, slice_num_z]
                    t2_slice = t2[:, :, :, slice_num_z]
                    label_slice = label[:, :, :, :, slice_num_z]
                    images_slice = torch.concat((flair_slice, t1_slice, t1ce_slice, t2_slice), dim=0).unsqueeze(0).to(self.device)
                    output = Unet_model(images_slice)
                    supervised_loss += criterion(output, label_slice)
                    predicted[:, :, :, slice_num_z] = output.detach()
                supervised_loss /= nums_z
                diceMetric.update(predicted.unsqueeze(0), label, supervised_loss.detach(), torch.tensor(0).to(self.device)) # maybe move to gpu to accelerate computing in the next training step
        dice_avg, supervised_loss_avg , _ = diceMetric.compute()
        dice_wl, dice_tc, dice_et = dice_avg
        dice_avg = torch.mean(dice_avg).cpu().detach().numpy()
        print(f"Model: {args.domain_source} Domain: {args.domain_source} Average Dice: {dice_avg} Whole Tumor: {dice_wl} Tumor Core: {dice_tc} Enhanced Tumor: {dice_et}")
        model.train()
        return dice_avg, supervised_loss_avg







