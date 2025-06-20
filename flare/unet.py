import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# --- 1. Basic Building Blocks (Refactored for Clarity) ---

class DoubleConv(nn.Module):
    """(Convolution => GroupNorm => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, dropout: float = 0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=mid_channels), # Using 8 groups is a common choice
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling then DoubleConv, with robust cropping"""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        # Use ConvTranspose2d to upsample and halve the number of channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Tensor from the previous layer (to be upsampled).
            x2: Tensor from the corresponding skip connection (to be cropped and concatenated).
        """
        x1 = self.up(x1)
        
        # --- THIS IS THE KEY FIX ---
        # Robust cropping for any input size.
        # Crop the skip connection tensor (x2) to match the spatial dimensions of the upsampled tensor (x1).
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x2 = F.pad(x2, [-diffX // 2, - (diffX - diffX // 2),
                        -diffY // 2, - (diffY - diffY // 2)])
        # -------------------------

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpBlockInterpolate(nn.Module):
    """Upscaling using interpolation, which avoids cropping entirely."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        # We don't need a ConvTranspose layer. We will use F.interpolate.
        # The input to DoubleConv will be the channel count from the skip
        # connection plus the channel count from the upsampled layer.
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Use interpolate to exactly match the size of the skip connection tensor
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
# --- 2. Main U-Net Architecture ---

class Unet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channel_list: List[int] = None, dropout_rates: List[float] = None):
        """
        A robust U-Net model that can handle arbitrary input sizes.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output classes.
            channel_list: List of channels for each level of the U-Net.
                          Default: [64, 128, 256, 512, 1024].
            dropout_rates: List of dropout rates for each level (4 down, 4 up).
                           Default: [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]
        """
        super(Unet, self).__init__()
        
        if channel_list is None:
            channel_list = [64, 128, 256, 512, 1024]
        if dropout_rates is None:
            # Dropout heavier in the bottleneck
            dropout_rates = [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]
            
        assert len(dropout_rates) == 8, "dropout_rates must have 8 values (4 down, 4 up)"

        self.in_conv = DoubleConv(in_channels, channel_list[0], dropout=dropout_rates[0])
        self.down1 = DownBlock(channel_list[0], channel_list[1], dropout=dropout_rates[1])
        self.down2 = DownBlock(channel_list[1], channel_list[2], dropout=dropout_rates[2])
        self.down3 = DownBlock(channel_list[2], channel_list[3], dropout=dropout_rates[3])
        
        # The downsampling factor determines the final downblock's input channels
        factor = 2 
        self.down4 = DownBlock(channel_list[3], channel_list[4] // factor, dropout=dropout_rates[4])

        self.up1 = UpBlockInterpolate(channel_list[4], channel_list[3] // factor, dropout=dropout_rates[5])
        self.up2 = UpBlockInterpolate(channel_list[3], channel_list[2] // factor, dropout=dropout_rates[6])
        self.up3 = UpBlockInterpolate(channel_list[2], channel_list[1] // factor, dropout=dropout_rates[7])
        self.up4 = UpBlockInterpolate(channel_list[1], channel_list[0]) # No dropout on last up block
        
        self.out_conv = nn.Conv2d(channel_list[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.out_conv(x)
        return logits

# --- 3. SimCLR Variant using the Robust Blocks ---
# Your original projector classes (can be kept as is or cleaned up)
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Projector(nn.Module):
    def __init__(self, in_channels: int, proj_hidden_dim: int, proj_output_dim: int):
        super().__init__()
        # Use Global Average Pooling to be robust to feature map size
        self.projector_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_channels, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector_head(x)

class Projector_CH_1(nn.Module):
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
        )

def forward(self, x):
    x = self.relu(self.instance_norm(self.conv1(x)))
    x = self.projector_layer(x)
    return x

class Unet_SegCLR(nn.Module):
    """A U-Net model with a SimCLR projection head on the encoder's bottleneck."""
    def __init__(self, in_channels: int, out_channels: int, proj_output_dim: int = 128):
        super().__init__()
        # We can reuse the standard Unet and just 'intercept' the forward pass
        self.unet = Unet(in_channels, out_channels)
        
        # The projector attaches to the bottleneck, which has channel_list[4] channels
        bottleneck_channels = 1024 # Based on default channel list
        self.projector = Projector(
            in_channels=bottleneck_channels // 2, 
            proj_hidden_dim=512, 
            proj_output_dim=proj_output_dim
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- Encoder Path ---
        x1 = self.unet.in_conv(x)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        bottleneck = self.unet.down4(x4)
        
        # --- Projection Head ---
        z = self.projector(bottleneck)
        
        # --- Decoder Path ---
        d1 = self.unet.up1(bottleneck, x4)
        d2 = self.unet.up2(d1, x3)
        d3 = self.unet.up3(d2, x2)
        d4 = self.unet.up4(d3, x1)
        
        logits = self.unet.out_conv(d4)
        
        return z, logits

# --- 4. Testing the Robustness ---
if __name__ == "__main__":
    # Test with a standard size (divisible by 16)
    print("--- Testing with standard size (256x256) ---")
    net = Unet(in_channels=1, out_channels=3)
    standard_input = torch.randn(2, 1, 256, 256)
    standard_output = net(standard_input)
    print(f"Input shape: {standard_input.shape}")
    print(f"Output shape: {standard_output.shape}\n") # Should be (2, 3, 256, 256)

    # Test with a non-standard, odd-numbered size
    print("--- Testing with non-standard size (251x387) ---")
    non_standard_input = torch.randn(2, 1, 251, 387)
    non_standard_output = net(non_standard_input)
    print(f"Input shape: {non_standard_input.shape}")
    print(f"Output shape: {non_standard_output.shape}") # Should be (2, 3, 251, 387)
    print("Test successful: Model is robust to arbitrary input sizes.")
    
    # Test SimCLR variant
    print("\n--- Testing Unet_SegCLR with non-standard size (251x387) ---")
    simclr_net = Unet_SegCLR(in_channels=1, out_channels=3)
    z, logits = simclr_net(non_standard_input)
    print(f"Projection Vector (z) shape: {z.shape}") # Should be (2, 128)
    print(f"Segmentation Logits shape: {logits.shape}") # Should be (2, 3, 251, 387)
    print("Test successful: Unet_SegCLR is also robust.")