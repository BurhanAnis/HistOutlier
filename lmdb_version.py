import argparse
import os
import pickle
import io

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import lmdb
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

# your autoencoder
from Dataset2 import ConvAutoencoder


class LMDBAugDataset(Dataset):
    """
    Reads patches (PNG bytes) + labels from an LMDB store.
    keys:     list of string keys (e.g. "slide123_000042")
    labels:   parallel list of 0/1 ints
    flags:    parallel list of bool: whether to apply aug_tf (True) or base_tf (False)
    """
    def __init__(self,
                 lmdb_path: str,
                 keys: list,
                 labels: list,
                 flags: list,
                 base_tf=None,
                 aug_tf=None):
        self.lmdb_path = lmdb_path
        self.keys       = [k.encode('ascii') for k in keys]
        self.labels     = labels
        self.flags      = flags
        self.base_tf    = base_tf or transforms.ToTensor()
        self.aug_tf     = aug_tf  or self.base_tf

        # will be initialized on first access in each worker
        self.env = None
        self.txn = None

    def _init_env(self):
        # readonly, no locks, so many workers can share safely
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=32,
        )
        self.txn = self.env.begin(buffers=False)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_env()

        data = self.txn.get(self.keys[idx])          # raw PNG bytes
        img  = Image.open(io.BytesIO(data)).convert('RGB')
        tf   = self.aug_tf if self.flags[idx] else self.base_tf
        img  = tf(img)

        return img, self.labels[idx]

    def __del__(self):
        if self.env is not None:
            self.env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train conv-autoencoder on WSI patches stored in LMDB"
    )
    parser.add_argument('--index_path', type=str, required=True,
                        help="Pickle of slide_index (same dict you used to build LMDB)")
    parser.add_argument('--lmdb_path',  type=str, required=True,
                        help="Directory of your LMDB environment")
    parser.add_argument('--aug_factor', type=int, default=50,
                        help="Extra augmented copies per positive patch")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--latent_dim',  type=int, default=128)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--epochs',      type=int, default=20,
                        help="Epochs of negative-only training")
    parser.add_argument('--output_dir',  type=str, default='output/')
    parser.add_argument('--plot',        action='store_true',
                        help="Save a t-SNE plot of test latents")
    parser.add_argument('--max_patches_train', type=int, default=None)
    parser.add_argument('--max_patches_test',  type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_file = os.path.join(args.output_dir, "latents_and_labels.npz")

    if os.path.exists(cache_file):
        print(f"Found cached {cache_file}, skipping training.")
        data = np.load(cache_file)
        latents_normal = data['latents_normal']
        latents_test   = data['latents_test']
        true_labels    = data['true_labels']
    else:
        # 1) load slide_index → reconstruct keys & labels in SAME order LMDB was built
        with open(args.index_path, 'rb') as f:
            slide_index = pickle.load(f)

        keys   = []
        labels = []
        count  = 0
        for slide_id, info in slide_index.items():
            for (y, x, is_tumor) in info['patches']:
                keys.append(f"{slide_id}_{count:06d}")
                labels.append(1 if is_tumor else 0)
                count += 1
        labels = np.array(labels, dtype=np.int64)

        # 2) stratified train/test split on flat indices
        N = len(labels)
        all_idx = np.arange(N)
        train_idx, test_idx = train_test_split(
            all_idx,
            test_size=0.5,
            stratify=labels,
            random_state=69
        )

        # optional trimming
        if args.max_patches_train is not None:
            train_idx = train_idx[:args.max_patches_train]
        if args.max_patches_test is not None:
            test_idx  = test_idx[:args.max_patches_test]

        # 3) build per-sample key/label/flag lists
        train_keys, train_labels, train_flags = [], [], []
        for idx in train_idx:
            train_keys.append(keys[idx])
            train_labels.append(int(labels[idx]))
            train_flags.append(False)
            if labels[idx] == 1:
                for _ in range(args.aug_factor):
                    train_keys.append(keys[idx])
                    train_labels.append(1)
                    train_flags.append(True)

        test_keys   = [keys[i] for i in test_idx]
        test_labels = [int(labels[i]) for i in test_idx]
        test_flags  = [False] * len(test_keys)

        # 4) transforms
        base_tf = transforms.Compose([
            transforms.ToTensor(),
        ])
        aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
        ])

        # 5) datasets & loaders
        train_ds = LMDBAugDataset(
            args.lmdb_path,
            train_keys,
            train_labels,
            train_flags,
            base_tf, aug_tf
        )
        test_ds = LMDBAugDataset(
            args.lmdb_path,
            test_keys,
            test_labels,
            test_flags,
            base_tf, base_tf
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # report
        total_test = len(test_ds)
        pos_test   = sum(1 for l in test_labels if l==1)
        neg_test   = total_test - pos_test
        print(f"Test set: {total_test} patches (neg={neg_test}, pos={pos_test})")
        print(f"Train set (with aug): {len(train_ds)} patches")

        # negative-only loader
        neg_positions = [i for i, l in enumerate(train_labels) if l == 0]
        neg_ds = Subset(train_ds, neg_positions)
        neg_loader = DataLoader(
            neg_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        # 6) model + optimizer
        device = (
            torch.device('mps') if torch.backends.mps.is_available()
            else torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )
        model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"→ Using DataParallel on GPUs: {model.device_ids}")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 7) negative-only training
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for imgs, _ in tqdm(neg_loader, desc=f"Epoch {epoch}/{args.epochs}"):
                imgs = imgs.to(device)
                optimizer.zero_grad()
                recon = model(imgs)
                loss  = criterion(recon, imgs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
            print(f"  Neg-only Loss: {running_loss / len(neg_ds):.6f}")

        # 8) extract latents for normals in train set
        real_model = model.module if isinstance(model, nn.DataParallel) else model
        real_model.eval()

        latents_normal = []
        for img, lbl in tqdm(train_ds, desc="Encoding normals"):
            if lbl != 0:
                continue
            with torch.no_grad():
                z = real_model.encoder(img.unsqueeze(0).to(device))
            latents_normal.append(z.cpu().squeeze().numpy())
        latents_normal = np.stack(latents_normal, axis=0)

        # 9) extract latents + labels on test set
        lat_list, lbl_list = [], []
        for imgs, lbls in tqdm(test_loader, desc="Encoding test"):
            imgs = imgs.to(device)
            with torch.no_grad():
                z = real_model.encoder(imgs)
            lat_list.append(z.cpu().numpy())
            lbl_list.append(lbls.numpy())

        latents_test = np.concatenate(lat_list, axis=0)
        true_labels  = np.concatenate(lbl_list, axis=0)

        # 10) save to cache
        np.savez(
            cache_file,
            latents_normal=latents_normal,
            latents_test=latents_test,
            true_labels=true_labels
        )

    # Optional t-SNE + scatter
    if args.plot:
        max_pts = 10000
        if len(true_labels) > max_pts:
            sel = np.random.choice(len(true_labels), max_pts, replace=False)
            data, lbls = latents_test[sel], true_labels[sel]
        else:
            data, lbls = latents_test, true_labels

        z2 = TSNE(n_components=2, random_state=42).fit_transform(data)
        plt.figure()
        plt.scatter(z2[lbls==0,0], z2[lbls==0,1], label='Normal', alpha=0.6)
        plt.scatter(z2[lbls==1,0], z2[lbls==1,1], label='Tumor',  alpha=0.6)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "tsne_plot.png"))
        print("→ Saved t-SNE plot.")
    

if __name__ == "__main__":
    main()
