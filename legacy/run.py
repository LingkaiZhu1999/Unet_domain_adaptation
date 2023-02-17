from model_2d_unet import UNet
from dataset import BratsTrainDataset
from torch.utils.data import DataLoader
import torch
from simCLR_unet_brats.loss.BCEDiceLoss import Loss
from torch.nn import init
import os

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
Epoch = 800
save_model_interval = 5
trained_model_dir = './model'
lr = 0.0003
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)

def train():
    train_dataset = BratsTrainDataset(datapath='/mnt/asgard2/data/lingkai/braTS20/slice', augmentation=None)
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=True)
    model = UNet().to(device)
    model.apply(weights_init_kaiming)
    lossDice = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0, last_epoch=-1)
    model.train()
    num = 0
    for epoch in range(Epoch):
        trainloss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = lossDice(output, target)

            loss.backward()

            optimizer.step()

            trainloss = trainloss + loss
            if (batch_idx + 1) % 5 == 0:
                trainloss = trainloss / 5
                print('train Epoch {} iteration: {} loss: {:.6f}'.format(epoch, batch_idx + 1, trainloss.data))
                trainloss = 0

            # if epoch >= 250: # warm up
            #     scheduler.step()
        if epoch % save_model_interval == 0:
            model_name = os.path.join(trained_model_dir, '{}.pkl'.format(epoch))
            torch.save(model.state_dict(), model_name)


train()
        



