import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}
ROTATABLE = {"bottle", "hazelnut", "metal_nut", "screw"}

def get_train_transform(category, size=256):
    aug = []
    if category in TEXTURES or category in ROTATABLE:
        aug += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.RandomApply([T.RandomRotation((90, 90))], p=0.5)]
    aug += [T.ColorJitter(brightness=0.05, contrast=0.05)]
    return T.Compose([T.Resize((size, size)), *aug, T.ToTensor()])

def get_test_transform(size=256):
    return T.Compose([T.Resize((size, size)), T.ToTensor()])

class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", size=256):
        self.split, self.size = split, size
        base = os.path.join(root, category)
        if split == "train":
            self.images = sorted(glob.glob(f"{base}/train/good/*.png"))
            self.transform = get_train_transform(category, size)
        else:
            self.images = sorted(glob.glob(f"{base}/test/*/*.png"))
            self.transform = get_test_transform(size)
        self.base = base

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        path = self.images[i]
        img = self.transform(Image.open(path).convert("RGB"))
        if self.split == "train":
            return img
        defect = path.split(os.sep)[-2]
        label = 0 if defect == "good" else 1
        if label:
            mpath = path.replace("test", "ground_truth").replace(".png", "_mask.png")
            mask = T.Compose([T.Resize((self.size, self.size)), T.ToTensor()])(Image.open(mpath))
        else:
            mask = torch.zeros(1, self.size, self.size)
        return img, label, (mask > 0.5).float()