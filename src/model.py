import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block with dilated convolutions for TCN.
    
    Dilated convolutions allow the network to have a large receptive field
    without increasing the number of parameters. This is crucial for astronomical
    light curves where long-term dependencies (e.g., rise/decay timescales) 
    must be captured efficiently.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # Padding calculation to maintain temporal length
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=padding, 
            dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size, 
            padding=padding, 
            dilation=dilation
        )
        self.relu2 = nn.ReLU()
        
        # Residual connection: if input/output channels differ, use 1x1 conv
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        # Add residual connection before final activation
        out = out + residual
        out = self.relu2(out)
        return out


class LightCurveTCN(nn.Module):
    """
    Temporal Convolutional Network for astronomical light curve classification.
    
    TCNs use dilated convolutions to efficiently capture long-range temporal 
    dependencies in irregular time series. This is essential for distinguishing
    between transient classes (SNe Ia vs. SNe II vs. AGN) where the characteristic
    timescale of brightness evolution is the key discriminator.
    
    Architecture:
    - Stacked residual blocks with exponentially increasing dilation rates
    - Each block doubles the receptive field, allowing the network to "see"
      patterns spanning the entire light curve without excessive parameters
    - Optimized for CPU inference (no batch normalization overhead)
    
    Args:
        n_channels: Number of input channels (filters: g-band, r-band)
        n_times: Temporal length of input (number of time steps)
        n_classes: Number of output classes for classification
    """
    def __init__(self, n_channels=2, n_times=100, n_classes=4):
        super().__init__()
        
        # TCN parameters
        hidden_channels = 32
        kernel_size = 3
        num_blocks = 4
        
        # Stack of residual blocks with increasing dilation
        # Dilation pattern: 1, 2, 4, 8 -> receptive field grows exponentially
        self.blocks = nn.ModuleList()
        in_ch = n_channels
        for i in range(num_blocks):
            dilation = 2 ** i  # Exponential dilation: captures multi-scale patterns
            self.blocks.append(
                ResidualBlock(in_ch, hidden_channels, kernel_size, dilation)
            )
            in_ch = hidden_channels
        
        # Global pooling aggregates temporal information
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        # Final classification layer
        self.fc = nn.Linear(hidden_channels, n_classes)
        
    def forward(self, x):
        # x shape: (batch, n_channels, n_times)
        # Pass through stacked TCN blocks
        for block in self.blocks:
            x = block(x)
        
        # Aggregate temporal dimension
        x = self.pool(x)       # (batch, hidden_channels, 1)
        x = self.flatten(x)    # (batch, hidden_channels)
        x = self.fc(x)         # (batch, n_classes)
        return x

