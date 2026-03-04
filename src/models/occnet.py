import torch
import torch.nn as nn

class SimpleOccNet(nn.Module):
    """
    Very simple CNN that maps a 1‑channel BEV occupancy grid [B,1,H,W]
    to a predicted occupancy probability grid [B,H,W].

    This is just a placeholder to demonstrate the DL pipeline.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W]
        returns: [B, H, W]
        """
        h = self.encoder(x)
        y = self.decoder(h)  # [B,1,H,W]
        return y.squeeze(1)