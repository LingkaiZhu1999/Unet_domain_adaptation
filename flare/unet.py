import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False

class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.dropout = dropout

    def forward(self, x):
        if self.dropout == True and self.training == True:
            x = F.dropout2d(F.relu(self.gn1(self.conv1(x))))
            y = F.dropout2d(F.relu(self.gn2(self.conv2(x))))
        else:
            x = F.relu(self.gn1(self.conv1(x)))
            y = F.relu(self.gn2(self.conv2(x)))
        x = F.max_pool2d(y, 2, stride=2)

        return x, y


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.dropout = dropout

    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        if self.dropout == True and self.training == True:
            x = F.dropout2d(F.relu(self.gn1(self.conv1(x))))
            x = F.dropout2d(F.relu(self.gn2(self.conv2(x))))
        else:
            x = F.relu(self.gn1(self.conv1(x)))
            x = F.relu(self.gn2(self.conv2(x)))

        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Projector_CH_1(nn.Module):
    """
    input: w * h * c
    output: 128
    """
    def __init__(self, in_channels, out_channels=1):
        super(Projector_CH_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.projector_layer = nn.Sequential(
            Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(64, 64)
            )

        ###worse performance ###
        # self.projector_layer = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(100, 100),
        #     nn.BatchNorm1d(100),
        #     nn.ReLU(),
        #     nn.Linear(100, 64),
        #     BatchNorm1dNoBias(64)
        #     )

    def forward(self, x):
        x = self.relu(self.instance_norm(self.conv1(x)))
        x = self.projector_layer(x)
        return x

class Projector_Cpool(nn.Module):
    """
    input: w * h * c
    output: 128
    """
    def __init__(self, in_channels=1024):
        super(Projector_Cpool, self).__init__()
        self.Cpool = nn.AvgPool2d(kernel_size=10)
        self.projector_layer = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 64),
            BatchNorm1dNoBias(64)
            )

    def forward(self, x):
        x = self.projector_layer(self.Cpool(x))
        return x

# class Projector(nn.Module):
#     """
#     input: w * h * c
#     output: 128
#     """
#     def __init__(self, in_channels, out_channels=1):
#         super(Projector, self).__init__()
#         self.projector_layer = nn.Sequential(
#             Flatten(),
#             nn.Linear(1024*10*10, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             )

#     def forward(self, x):
#         x = self.projector_layer(x)
#         return x


class Unet_dropout(nn.Module):
    def __init__(self, in_channel=4, out_channel=3, dropout=True):
        super(Unet_dropout, self).__init__()
        self.down1 = Downsample_block(in_channel, 64, False)
        self.down2 = Downsample_block(64, 128, False)
        self.down3 = Downsample_block(128, 256, dropout)
        self.down4 = Downsample_block(256, 512, dropout)
        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=1024)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=1024)
        self.up4 = Upsample_block(1024, 512, dropout) # reason is here
        self.up3 = Upsample_block(512, 256, dropout)
        self.up2 = Upsample_block(256, 128, False)
        self.up1 = Upsample_block(128, 64, False)
        self.outconv = nn.Conv2d(64, out_channel, 1)
        self.dropout = dropout

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        if self.training and self.dropout:
            x = F.dropout2d(F.relu(self.gn1(self.conv1(x))))
            x = F.dropout2d(F.relu(self.gn2(self.conv2(x)))) 
        else:
            x = F.relu(self.gn1(self.conv1(x)))
            x = F.relu(self.gn2(self.conv2(x))) 
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)
        return x1

class Unet(nn.Module):
    def __init__(self, in_channel=4, out_channel=3, dropout=False):
        super(Unet, self).__init__()
        self.down1 = Downsample_block(in_channel, 64)
        self.down2 = Downsample_block(64, 128)
        self.down3 = Downsample_block(128, 256)
        self.down4 = Downsample_block(256, 512)
        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=1024)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=1024)
        self.up4 = Upsample_block(1024, 512, True) # reason is here
        self.up3 = Upsample_block(512, 256, True)
        self.up2 = Upsample_block(256, 128)
        self.up1 = Upsample_block(128, 64)
        self.outconv = nn.Conv2d(64, out_channel, 1)
        self.dropout = dropout

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        if self.training and self.dropout:
            x = F.dropout2d(F.relu(self.gn1(self.conv1(x))))
            x = F.dropout2d(F.relu(self.gn2(self.conv2(x)))) 
        else:
            x = F.relu(self.gn1(self.conv1(x)))
            x = F.relu(self.gn2(self.conv2(x))) 
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)
        return x1


class Unet_SegCLR(nn.Module):
    def __init__(self, in_channel=4, out_channel=3, dropout=False):
        super(Unet_SegCLR, self).__init__()
        self.down1 = Downsample_block(in_channel, 64)
        self.down2 = Downsample_block(64, 128)
        self.down3 = Downsample_block(128, 256)
        self.down4 = Downsample_block(256, 512)
        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=1024)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=1024)
        self.up4 = Upsample_block(1024, 512, dropout)
        self.up3 = Upsample_block(512, 256, dropout)
        self.up2 = Upsample_block(256, 128)
        self.up1 = Upsample_block(128, 64)
        self.outconv = nn.Conv2d(64, out_channel, 1)
        self.projector = Projector_CH_1(in_channels=1024)

    def forward(self, x, only_encoder=False):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        # x = F.dropout2d(F.relu(self.gn1(self.conv1(x))))
        x = F.relu(self.gn1(self.conv1(x)))
        z = self.projector(x)
        if not only_encoder:
            # x = F.dropout2d(F.relu(self.gn2(self.conv2(x))))
            x = F.relu(self.gn2(self.conv2(x)))
            x = self.up4(x, y4)
            x = self.up3(x, y3)
            x = self.up2(x, y2)
            x = self.up1(x, y1)
            x1 = self.outconv(x)
            return z, x1
        else:
            return z

class Encoder_SimCLR(nn.Module):
    def __init__(self, in_channel=4):
        super(Encoder_SimCLR, self).__init__()
        self.down1 = Downsample_block(in_channel, 64)
        self.down2 = Downsample_block(64, 128)
        self.down3 = Downsample_block(128, 256)
        self.down4 = Downsample_block(256, 512)
        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=1024)
        self.projector = Projector_CH_1(in_channels=1024)

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        # x = F.dropout2d(F.relu(self.gn1(self.conv1(x))))
        x = F.relu(self.gn1(self.conv1(x)))
        z = self.projector(x)
        return z

if __name__ == "__main__":
    unet = Unet(in_channel=4, out_channel=3)
    x = torch.randn(1, 4, 240, 240)
    y = unet(x)
    print(y)