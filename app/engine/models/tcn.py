from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple


class Chomp1d(nn.Module):
    """
    Enforces strict temporal causality by removing padding added to the future end of the sequence.
    """
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slices off the trailing future padding along the temporal dimension
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Dilated Causal Residual Block:
    Two layers of dilated causal convolutions + weight normalization + ReLU + residual connection.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Conv Layer 2
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual projection if channel dimensions differ
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1: Dilated causal convolution path
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Branch 2: Identity / 1x1 conv skip connection
        residual = x if self.downsample is None else self.downsample(x)

        return self.relu_out(out + residual)


class DilatedCausalTCN(nn.Module):
    """
    Dual-Head Crucible Diagnostics Engine:
    1. Dilated Causal 1D-TCN backbone over reconstructed phase space (25 channels).
    2. Multi-class classification head: 3 degradation classes (Softmax).
    3. Continuous regression head: Dynamic lead time to failure in hours (ReLU).
    """
    def __init__(
        self,
        in_channels: int = 25,
        num_classes: int = 3,
        num_channels: Tuple[int, ...] = (32, 64, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super(DilatedCausalTCN, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation: 1, 2, 4, 8
            input_dim = in_channels if i == 0 else num_channels[i - 1]
            output_dim = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)
        last_channel_dim = num_channels[-1]

        # Global temporal pooling to summarize sequence trajectory
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Head 1: 3-Class Degradation Classifier (Class A, B, C)
        self.classifier_head = nn.Sequential(
            nn.Linear(last_channel_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        # Head 2: Dynamic Lead-Time-to-Failure (Hours remaining)
        self.lead_time_head = nn.Sequential(
            nn.Linear(last_channel_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # Enforces strictly non-negative remaining hours
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x shape: (Batch, in_channels=25, sequence_length)
        Returns:
            class_logits: shape (Batch, 3)
            lead_time_hours: shape (Batch, 1)
        """
        # Pass through dilated causal backbone
        features = self.network(x)  # Shape: (Batch, 128, sequence_length)

        # Pool temporal dimension
        pooled = self.global_pool(features).squeeze(-1)  # Shape: (Batch, 128)

        # Dual forward heads
        class_logits = self.classifier_head(pooled)
        lead_time = self.lead_time_head(pooled)

        return class_logits, lead_time
    
from pathlib import Path

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "crucible_tcn_v1.pt"


def load_tcn_model(weights_path: Path = DEFAULT_WEIGHTS_PATH) -> DilatedCausalTCN:
        """
        Factory that instantiates DilatedCausalTCN and loads state dict from disk.
        Ensures safe deserialization via weights_only=True.
        """
        in_channels = 25
        num_classes = 3
        num_channels = (32, 64, 64, 128)

        if weights_path.exists():
            checkpoint = torch.load(weights_path, map_location=torch.device("cpu"), weights_only=True)
            model = DilatedCausalTCN(
                in_channels=checkpoint.get("in_channels", in_channels),
                num_classes=checkpoint.get("num_classes", num_classes),
                num_channels=checkpoint.get("num_channels", num_channels),
                kernel_size=checkpoint.get("kernel_size", 3),
                dropout=checkpoint.get("dropout", 0.0),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = DilatedCausalTCN(
                in_channels=in_channels,
                num_classes=num_classes,
                num_channels=num_channels,
            )

        model.eval()
        return model