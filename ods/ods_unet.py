import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class Projector_CH(nn.Module):
    """
    input: w * h * c
    output: 128
    """
    def __init__(self, in_channels, out_channels=1):
        super(Projector_CH, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.projector_layer = nn.Sequential(
            Flatten(),
            nn.Linear(36*36, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
            )

    def forward(self, x):
        x = self.relu(self.instance_norm(self.conv1(x)))
        x = self.projector_layer(x)
        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, in_channels=3, classes=1):
        super(U_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, classes, kernel_size=1, stride=1, padding=0)
        # for the bottleneck level's projector
        self.bottleneck_conv1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.batchnorm2d = nn.BatchNorm2d(num_features=1024)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck_conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
    
    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x) 

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2) 
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3) 

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4) 

        x5 = self.Maxpool(x4)

        # x5 = self.Conv5(x5)
        x5 = self.relu(self.batchnorm2d(self.bottleneck_conv1(x5)))
        x5 = self.relu(self.batchnorm2d(self.bottleneck_conv2(x5)))
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5), dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1 
        
class SegCLR_U_Net(nn.Module):
    def __init__(self, in_channels=1, classes=1):
        super(SegCLR_U_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, classes, kernel_size=1, stride=1, padding=0)
        # for the bottleneck level's projector
        self.bottleneck_conv1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.batchnorm2d = nn.BatchNorm2d(num_features=1024)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck_conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.projector = Projector_CH(in_channels=1024)
    
    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def forward(self, x, only_encoder=False):
        # encoding path
        x1 = self.Conv1(x) 

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2) 
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3) 

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4) 

        x5 = self.Maxpool(x4)

        # x5 = self.Conv5(x5)
        x5 = self.relu(self.batchnorm2d(self.bottleneck_conv1(x5)))
        project_head = self.projector(x5)
        if not only_encoder:
            x5 = self.relu(self.batchnorm2d(self.bottleneck_conv2(x5)))
            # decoding + concat path
            d5 = self.Up5(x5)
            d5 = torch.cat((x4,d5), dim=1)
            
            d5 = self.Up_conv5(d5)
            
            d4 = self.Up4(d5)
            d4 = torch.cat((x3,d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((x2,d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((x1,d2), dim=1)
            d2 = self.Up_conv2(d2)

            d1 = self.Conv_1x1(d2)

            return project_head, d1
        else:
            return project_head