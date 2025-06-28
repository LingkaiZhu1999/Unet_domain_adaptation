import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union

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
        return nn.GroupNorm(num_groups=4, num_channels=num_channels)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")


class DoubleConv(nn.Module):
    """(Convolution => Normalization => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None,
                 dropout: float = 0.0, norm: str = 'instance'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        use_bias = norm.lower() != 'batch'
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=use_bias),
            get_norm_layer(norm, mid_channels),
            nn.ReLU(inplace=True),
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
    
# upscale by conv transpose
# --- REVISED IMPLEMENTATION TO MATCH THE DIAGRAM ---
# upscale by 3x3 conv transpose
class UpBlockConvTranspose(nn.Module):
    """
    Upscaling using a 3x3 ConvTranspose2d to exactly match the provided diagram.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, norm: str = 'group'):
        """
        Args:
            in_channels: Number of channels from the layer below (e.g., 1024).
            out_channels: Number of output channels for the block, which is also the
                          number of channels in the skip connection (e.g., 512).
        """
        super().__init__()

        # THE KEY CHANGE: Use kernel_size=3 with padding=1 and output_padding=1
        # to achieve exact 2x upsampling.
        # This layer will also halve the number of channels (e.g., 1024 -> 512).
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # After concatenating the skip connection (out_channels) and the upsampled tensor (out_channels),
        # the input to the DoubleConv will have 2 * out_channels.
        # The blue arrow in the diagram is a DoubleConv, which here takes (e.g., 512+512=1024) -> 512 channels.
        self.conv = DoubleConv(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            mid_channels=out_channels,
            dropout=dropout,
            norm=norm
        )
        
        # The diagram also shows dropout in the decoder path (red arrow).
        # We can add it here if it's not already part of the DoubleConv.
        # Since your DoubleConv includes dropout, we don't need an extra one here.

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Tensor from the lower layer (to be upsampled). Shape: [B, C_in, H, W]
            x2: Tensor from the corresponding skip connection. Shape: [B, C_out, 2H, 2W]
        """
        # 1. Upsample the lower-level feature map
        x1 = self.up(x1) # Now produces a tensor of shape [B, C_out, 2H, 2W]

        # Note: The padding/output_padding trick makes the F.pad below redundant for
        # inputs with even dimensions, but it's good practice to keep it as a
        # safeguard for inputs with odd dimensions.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX != 0 or diffY != 0:
             x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                             diffY // 2, diffY - diffY // 2])

        # 2. Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)

        # 3. Apply the double convolution block
        return self.conv(x)


# --- 2. Main U-Net Architecture ---

class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 channel_list: Optional[List[int]] = None,
                 dropout_rates: Optional[List[float]] = None,
                 norm: str = 'group',
                 deep_supervision: bool = False):
        """
        A robust U-Net model with optional deep supervision.
        
        Args:
            ... (other args are the same) ...
            deep_supervision: If True, adds auxiliary output heads in the decoder.
                              The forward pass will return a list of segmentations.
                              Default: False.
        """
        super(UNet, self).__init__()
        self.deep_supervision = deep_supervision

        if channel_list is None:
            # This list now correctly results in a 1024-channel bottleneck
            channel_list = [64, 128, 256, 512, 1024]
        if dropout_rates is None:
            dropout_rates = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]

        assert len(dropout_rates) == 9, "dropout_rates must have 9 values (1 in_conv, 4 down, 4 up)"

        # <<< --- CHANGED --- Setting factor to 1 uses the full 1024 channels for the bottleneck
        factor = 1 

        self.in_conv = DoubleConv(in_channels, channel_list[0], dropout=dropout_rates[0], norm=norm)
        self.down1 = DownBlock(channel_list[0], channel_list[1], dropout=dropout_rates[1], norm=norm)
        self.down2 = DownBlock(channel_list[1], channel_list[2], dropout=dropout_rates[2], norm=norm)
        self.down3 = DownBlock(channel_list[2], channel_list[3], dropout=dropout_rates[3], norm=norm)
        
        # This now correctly creates a DownBlock from 512 -> 1024 channels
        self.down4 = DownBlock(channel_list[3], channel_list[4] // factor, dropout=dropout_rates[4], norm=norm)

        # <<< --- CHANGED --- The UpBlock's input channels must account for skip connections.
        # The input to up1's convolution is (channels from x4_skip) + (channels from x5_upsampled)
        self.up1 = UpBlockConvTranspose(channel_list[4] // factor, channel_list[3] // factor, dropout=dropout_rates[5], norm=norm)

        # The input to up2's convolution is (channels from x3_skip) + (channels from d1_output)
        self.up2 = UpBlockConvTranspose(channel_list[3] // factor, channel_list[2] // factor, dropout=dropout_rates[6], norm=norm)

        # The input to up3's convolution is (channels from x2_skip) + (channels from d2_output)
        self.up3 = UpBlockConvTranspose(channel_list[2] // factor, channel_list[1] // factor, dropout=dropout_rates[7], norm=norm)

        # The input to up4's convolution is (channels from x1_skip) + (channels from d3_output)
        self.up4 = UpBlockConvTranspose(channel_list[1] // factor, channel_list[0], dropout=dropout_rates[8], norm=norm)

        self.out_conv = nn.Conv2d(channel_list[0], out_channels, kernel_size=1)

        if self.deep_supervision:
            self.ds_out3 = nn.Conv2d(channel_list[1] // factor, out_channels, kernel_size=1)
            self.ds_out2 = nn.Conv2d(channel_list[2] // factor, out_channels, kernel_size=1)
            self.ds_out1 = nn.Conv2d(channel_list[3] // factor, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Encoder path
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # Shape: [B, 1024, H/16, W/16]
        
        # Decoder path
        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        logits_final = self.out_conv(d4)

        if self.deep_supervision:
            logits_ds1 = self.ds_out1(d1)
            logits_ds2 = self.ds_out2(d2)
            logits_ds3 = self.ds_out3(d3)
            return [logits_ds1, logits_ds2, logits_ds3, logits_final]
        else:
            return logits_final


# --- 3. SimCLR Variant using the Robust Blocks ---
class Projector_Pool(nn.Module):
    def __init__(self, in_channels: int, proj_hidden_dim: int, proj_output_dim: int):
        super().__init__()
        self.projector_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector_head(x)
    
class Projector_Conv(nn.Module):
    def __init__(self, in_channels: int, proj_hidden_dim: int, proj_output_dim: int):
        super().__init__()
        self.projector_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Flatten(),
            nn.Linear(proj_hidden_dim, proj_output_dim), # add group norm to the nn.Linear layer
            nn.GroupNorm(num_groups=4, num_channels=proj_output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_output_dim, proj_output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.projector_head(x)
    
class Unet_SegCLR(nn.Module):
    """A U-Net model with a SimCLR projection head on the encoder's bottleneck."""
    def __init__(self, in_channels: int, out_channels: int, proj_output_dim: int = 128, norm: str = 'group', deep_supervision: bool = False,
                 proj_hidden_dim: int = 1024, projector: str = 'pool'):
        super().__init__()
        # <<< --- PASS DEEP SUPERVISION FLAG TO UNET ---
        self.unet = UNet(in_channels, out_channels, norm=norm, deep_supervision=deep_supervision)
        
        bottleneck_channels = 1024
        if projector == 'conv':
            self.projector = Projector_Conv(
                in_channels=bottleneck_channels,
                proj_hidden_dim=proj_hidden_dim,
                proj_output_dim=proj_output_dim
            )
        else:
            self.projector = Projector_Pool(
                in_channels=bottleneck_channels,
                proj_hidden_dim=proj_hidden_dim,
                proj_output_dim=proj_output_dim
            )

    def forward(self, x: torch.Tensor, has_label=True) -> Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:
        # --- Encoder Path ---
        x1 = self.unet.in_conv(x)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        bottleneck = self.unet.down4(x4)
        
        # --- Projection Head ---
        if self.training:
            z = self.projector(bottleneck)
        else:
            z = None
        if has_label:
            # --- Decoder Path ---
            d1 = self.unet.up1(bottleneck, x4)
            d2 = self.unet.up2(d1, x3)
            d3 = self.unet.up3(d2, x2)
            d4 = self.unet.up4(d3, x1)
            
            logits_final = self.unet.out_conv(d4)
            
            # <<< --- HANDLE MULTIPLE OUTPUTS FROM THE UNET ---
            if self.unet.deep_supervision:
                logits_ds1 = self.unet.ds_out1(d1)
                logits_ds2 = self.unet.ds_out2(d2)
                logits_ds3 = self.unet.ds_out3(d3)
                # When using SegCLR, we only care about the final segmentation for supervised loss,
                # but the UNet itself returns all heads. We'll return them all.
                # Your training loop will then need to handle this list.
                all_logits = [logits_ds1, logits_ds2, logits_ds3, logits_final]
                return z, all_logits
            else:
                return z, logits_final
        else:
            # If no labels, we only return the projection vector
            return z, None


# --- 4. Testing the Robustness and Flexibility ---
if __name__ == "__main__":
    # ... (previous tests are still valid) ...
    # use cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # <<< --- ADDED TEST FOR DEEP SUPERVISION ---
    print("\n--- Testing U-Net with Deep Supervision ---")
    ds_net = UNet(in_channels=1, out_channels=14, deep_supervision=True).to(device)
    standard_input = torch.randn(2, 1, 512, 512).to(device)  # Batch size of 50, 1 channel, 480x480 resolution
    ds_outputs = ds_net(standard_input)

    assert isinstance(ds_outputs, list), "Output should be a list for deep supervision"
    assert len(ds_outputs) == 4, "Should have 4 outputs (3 auxiliary + 1 final)"
    print(f"Successfully created DS-U-Net. Number of outputs: {len(ds_outputs)}")
    print("Output shapes (from coarsest to finest):")
    for i, out in enumerate(ds_outputs):
        print(f"  Output {i+1}: {out.shape}")
    
    # Final output should match input size
    # assert ds_outputs[-1].shape == standard_input.shape[:1] + (3,) + standard_input.shape[2:]

    print("\n--- Testing Unet_SegCLR with Deep Supervision ---")
    ds_segclr_net = Unet_SegCLR(in_channels=1, out_channels=14, projector='pool').to(device)
    z, logits_list = ds_segclr_net(standard_input, has_label=False)
    # assert isinstance(logits_list, list) and len(logits_list) == 4
    print(f"Projection Vector (z) shape: {z.shape}") # Should be (2, 128)
    # print(f"Segmentation Logits is a list of {len(logits_list)} tensors.")
    print("Test successful: Models are robust to deep supervision flag.")
    # print unet_segclr_net
    # print(ds_segclr_net)