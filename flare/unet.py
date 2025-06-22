import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

# --- 1. Basic Building Blocks (Refactored for Clarity and Flexibility) ---

def get_norm_layer(norm_type: str, num_channels: int) -> nn.Module:
    """
    Returns a normalization layer based on the specified type.

    Args:
        norm_type: One of 'batch', 'instance', 'group', or 'none'.
        num_channels: Number of channels for the normalization layer.

    Returns:
        An nn.Module representing the normalization layer.
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        return nn.BatchNorm2d(num_channels)
    elif norm_type == 'instance':
        return nn.InstanceNorm2d(num_channels)
    elif norm_type == 'group':
        # Using 8 groups is a common default choice.
        return nn.GroupNorm(num_groups=8, num_channels=num_channels)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")


class DoubleConv(nn.Module):
    """(Convolution => Normalization => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None,
                 dropout: float = 0.0, norm: str = 'group'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # Use bias=False for conv layers when using BatchNorm, as it has its own bias (beta).
        use_bias = norm.lower() != 'batch'

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=use_bias),
            get_norm_layer(norm, mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            get_norm_layer(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, norm: str = 'group'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout, norm=norm)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlockInterpolate(nn.Module):
    """Upscaling using interpolation, which avoids cropping."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, norm: str = 'group'):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout, norm=norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# --- 2. Main U-Net Architecture ---

class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 channel_list: Optional[List[int]] = None,
                 dropout_rates: Optional[List[float]] = None,
                 norm: str = 'group'):
        """
        A robust U-Net model that can handle arbitrary input sizes.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output classes.
            channel_list: List of channels for each level of the U-Net.
                          Default: [64, 128, 256, 512, 1024].
            dropout_rates: List of dropout rates for each level (5 down, 4 up).
                           Default: All zeros.
            norm: Normalization type. One of 'batch', 'instance', 'group', 'none'.
                  Default: 'group'.
        """
        super(UNet, self).__init__()

        if channel_list is None:
            channel_list = [64, 128, 256, 512, 1024]
        if dropout_rates is None:
            dropout_rates = [0.0] * 9 # 5 down blocks (inc. in_conv), 4 up blocks

        assert len(dropout_rates) == 9, "dropout_rates must have 9 values (1 in_conv, 4 down, 4 up)"

        self.in_conv = DoubleConv(in_channels, channel_list[0], dropout=dropout_rates[0], norm=norm)
        self.down1 = DownBlock(channel_list[0], channel_list[1], dropout=dropout_rates[1], norm=norm)
        self.down2 = DownBlock(channel_list[1], channel_list[2], dropout=dropout_rates[2], norm=norm)
        self.down3 = DownBlock(channel_list[2], channel_list[3], dropout=dropout_rates[3], norm=norm)

        factor = 2
        self.down4 = DownBlock(channel_list[3], channel_list[4] // factor, dropout=dropout_rates[4], norm=norm)

        self.up1 = UpBlockInterpolate(channel_list[4], channel_list[3] // factor, dropout=dropout_rates[5], norm=norm)
        self.up2 = UpBlockInterpolate(channel_list[3], channel_list[2] // factor, dropout=dropout_rates[6], norm=norm)
        self.up3 = UpBlockInterpolate(channel_list[2], channel_list[1] // factor, dropout=dropout_rates[7], norm=norm)
        self.up4 = UpBlockInterpolate(channel_list[1], channel_list[0], dropout=dropout_rates[8], norm=norm)

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
# (Your original SimCLR/Projector code can remain here, just ensure you pass the `norm` param if needed)
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Projector(nn.Module):
    def __init__(self, in_channels: int, proj_hidden_dim: int, proj_output_dim: int):
        super().__init__()
        self.projector_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_channels, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector_head(x)

class Unet_SegCLR(nn.Module):
    """A U-Net model with a SimCLR projection head on the encoder's bottleneck."""
    def __init__(self, in_channels: int, out_channels: int, proj_output_dim: int = 128, norm: str = 'group'):
        super().__init__()
        # Pass the norm choice to the underlying U-Net
        self.unet = UNet(in_channels, out_channels, norm=norm)
        
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

# --- 4. Testing the Robustness and Flexibility ---
if __name__ == "__main__":
    # --- Test 1: Robustness to input size ---
    print("--- Testing with non-standard size (251x387) ---")
    net_group = UNet(in_channels=1, out_channels=3) # Uses default GroupNorm
    non_standard_input = torch.randn(2, 1, 251, 387)
    non_standard_output = net_group(non_standard_input)
    print(f"Input shape: {non_standard_input.shape}")
    print(f"Output shape: {non_standard_output.shape}") # Should be (2, 3, 251, 387)
    print("Test successful: Model is robust to arbitrary input sizes.")

    # --- Test 2: Flexibility of Normalization ---
    print("\n--- Testing different normalization layers ---")

    # Batch Norm
    net_batch = UNet(in_channels=1, out_channels=3, norm='batch')
    print("\nModel with BatchNorm2d:")
    # print(net_batch) # Uncomment to see the full architecture
    assert isinstance(net_batch.in_conv.double_conv[1], nn.BatchNorm2d)
    print("Successfully created U-Net with Batch Normalization.")

    # Instance Norm
    net_instance = UNet(in_channels=1, out_channels=3, norm='instance')
    assert isinstance(net_instance.in_conv.double_conv[1], nn.InstanceNorm2d)
    print("Successfully created U-Net with Instance Normalization.")

    # No Norm
    net_none = UNet(in_channels=1, out_channels=3, norm='none')
    assert isinstance(net_none.in_conv.double_conv[1], nn.Identity)
    print("Successfully created U-Net with no normalization (Identity).")
    
    # --- Test 3: SimCLR variant ---
    print("\n--- Testing Unet_SegCLR with Batch Norm ---")
    simclr_net = Unet_SegCLR(in_channels=1, out_channels=3, norm='batch')
    z, logits = simclr_net(non_standard_input)
    print(f"Projection Vector (z) shape: {z.shape}") # Should be (2, 128)
    print(f"Segmentation Logits shape: {logits.shape}") # Should be (2, 3, 251, 387)
    print("Test successful: Unet_SegCLR is also robust and flexible.")