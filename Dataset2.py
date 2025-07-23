import pickle
from torch.utils.data import Dataset, DataLoader
import openslide
from torchvision import transforms
import torch
import torch.nn as nn


class WSIDataset(Dataset):
    def __init__(self, patches, base_transform = None, aug_transform = None):

        self.patches = patches
        self.base_tf = base_transform or transforms.ToTensor()
        self.aug_tf = aug_transform or self.base_tf

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, i):

        info = self.patches[i]
        slide = openslide.OpenSlide(info['slide_path'])

        lvl = info['level']                   # e.g. 4
        down = slide.level_downsamples[lvl]   # e.g. 16

        # coords are in level-lvl space; convert to level-0:
        x_lvl, y_lvl = info['coords']
        x0 = int(x_lvl * down)
        y0 = int(y_lvl * down)

        patch = slide.read_region((x0, y0), lvl, (256,256)).convert('RGB')
        slide.close()

        tf = self.aug_tf if info.get('augment', False) else self.base_tf
        img = tf(patch)
        label = 1 if info['is_tumor'] else 0
        return img, label 
    


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder: 3×256×256 → latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 32×128×128
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64×64×64
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 128×32×32
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 256×16×16
            nn.ReLU(True),
            nn.Flatten(),                                # 256*16*16
            nn.Linear(256*16*16, latent_dim)
        )
        # Decoder: latent_dim → 3×256×256
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256*16*16),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 128×32×32
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64×64×64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32×128×128
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 3×256×256
            nn.Sigmoid()  # outputs in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
