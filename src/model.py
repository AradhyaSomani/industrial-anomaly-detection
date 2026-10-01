import torch.nn as nn

def down(i, o): return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
def up(i, o):   return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())

class ConvAE(nn.Module):
    def __init__(self, latent_ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
            down(3, 32), down(32, 64), down(64, 128), down(128, 256),   # 256 → 16
            nn.Conv2d(256, latent_ch, 3, 1, 1)                           # bottleneck: latent_ch × 16 × 16
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, 256, 3, 1, 1), nn.ReLU(),
            up(256, 128), up(128, 64), up(64, 32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.decoder(self.encoder(x))