import torch, torch.nn as nn, torch.nn.functional as F, cv2
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

class FeatureExtractor(nn.Module):
    """Frozen ImageNet WideResNet-50: layer2 + layer3 features at 32x32."""
    def __init__(self):
        super().__init__()
        m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.layer2, self.layer3 = m.layer2, m.layer3
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters(): p.requires_grad = False
        self.eval()

    def train(self, mode=True):          # never leave eval mode (keeps BatchNorm frozen)
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        x = (x - self.mean) / self.std
        f2 = self.layer2(self.stem(x))                     # 512 x 32 x 32
        f3 = F.interpolate(self.layer3(f2), size=f2.shape[-2:],
                           mode="bilinear", align_corners=False)  # 1024 x 32 x 32
        return F.avg_pool2d(torch.cat([f2, f3], 1), 3, 1, 1)      # 1536 x 32 x 32, local context

class FeatureAE(nn.Module):
    """1x1-conv autoencoder over feature vectors."""
    def __init__(self, in_ch=1536, latent=200):
        super().__init__()
        mid = (in_ch + latent) // 2
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.encoder = nn.Sequential(blk(in_ch, mid), blk(mid, 2 * latent), nn.Conv2d(2 * latent, latent, 1))
        self.decoder = nn.Sequential(blk(latent, 2 * latent), blk(2 * latent, mid), nn.Conv2d(mid, in_ch, 1))

    def forward(self, f): return self.decoder(self.encoder(f))

@torch.no_grad()
def feature_anomaly_map(ext, ae, x, device, out_size=256, sigma=4):
    f = ext(x.unsqueeze(0).to(device))
    err = ((f - ae(f)) ** 2).mean(1, keepdim=True)                 # 1 x 1 x 32 x 32
    err = F.interpolate(err, size=out_size, mode="bilinear", align_corners=False)
    return cv2.GaussianBlur(err[0, 0].cpu().numpy(), (0, 0), sigma)

@torch.no_grad()
def map_from_features(f, ae, out_size=256, sigma=4):
    err = ((f - ae(f)) ** 2).mean(1, keepdim=True)
    err = F.interpolate(err, size=out_size, mode="bilinear", align_corners=False)
    return cv2.GaussianBlur(err[0, 0].cpu().numpy(), (0, 0), sigma)