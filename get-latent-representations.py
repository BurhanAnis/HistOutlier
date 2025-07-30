from Dataset2 import WSIDataset, ConvAutoencoder
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader, Subset
import torch
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate latent representations for WSI patches.")
parser.add_argument('--index_path', type=str, required=True,
                    help='Path to the index-level pickle file')
parser.add_argument('--aug_factor', type=int, default=50,
                    help='Number of augmented copies per positive sample')
parser.add_argument('--num_workers', type = int, default = 4)
parser.add_argument('--latent_dim', type = int, default = 128)
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for DataLoader')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs for neg-only training')
parser.add_argument('--output_dir', type=str, default = 'output/')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--max_patches_train', type = int, default = None)
parser.add_argument('--max_patches_test', type = int, default = None)
args = parser.parse_args()

# Assign arguments to variables
index_path = args.index_path
AUG_FACTOR = args.aug_factor
num_workers = args.num_workers
batch_size = args.batch_size
n_epochs = args.epochs
latent_dim = args.latent_dim
plot = args.plot

output_path = os.path.join(args.output_dir, "latents_and_labels.npz")
os.makedirs(args.output_dir, exist_ok=True)

if os.path.exists(output_path):
    print(f"Found cached latent file at {output_path}. Skipping model training and extraction.")
    data = np.load(output_path)
    latents_normal = data["latents_normal"]
    latents_test = data["latents_test"]
    true_labels = data["true_labels"]

else:


    with open(index_path, 'rb') as f:
        slide_index = pickle.load(f)

    all_patches = []
    for slide_info in slide_index.values():
        for y, x, is_tumor in slide_info['patches']:
            all_patches.append({
                'slide_path': slide_info['slide_path'],
                'coords': (x, y),
                'level': slide_info['level'],
                'is_tumor': is_tumor
            })

    labels = [1 if p['is_tumor'] else 0 for p in all_patches]
    train_raw, test_raw = train_test_split(
        all_patches,
        test_size=0.5,
        stratify=labels,
        random_state=69

    )

    if args.max_patches_train is not None:
        train_raw = train_raw[: args.max_patches_train]
    if args.max_patches_test is not None:
        test_raw  = test_raw[:  args.max_patches_test]


    AUG_FACTOR = AUG_FACTOR   # how many extra rotated/flipped copies per positive
    train_augmented = []
    for p in train_raw:
        # always keep the “original” (no augment)
        train_augmented.append({**p, 'augment': False})

        # if tumor, add N augmented copies
        if p['is_tumor']:
            for _ in range(AUG_FACTOR):
                train_augmented.append({**p, 'augment': True})

    base_transform = transforms.ToTensor()
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor()
    ])

    train_ds = WSIDataset(train_augmented, base_transform, aug_transform)
    test_ds  = WSIDataset(test_raw,       base_transform, base_transform)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=num_workers)

    total = len(test_ds.patches)
    positives = sum(1 for p in test_ds.patches if p['is_tumor'])
    negatives = total - positives

    print(f'Test set: {total} patches')
    print(f'  ➤ Negatives (label 0): {negatives}')
    print(f'  ➤ Positives (label 1): {positives}')
    print(f'Training patches size: {len(train_ds)} patches')

    neg_indices = [
        i
        for i, p in enumerate(train_ds.patches)
        if not p['is_tumor']      # p['is_tumor'] == False → negative
    ]

    neg_ds     = Subset(train_ds, neg_indices)
    neg_loader = DataLoader(
        neg_ds,
        batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )


    # Set up model, loss, optimizer as before
    device = (
        torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu')
    )
    model     = ConvAutoencoder(latent_dim=latent_dim).to(device)

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"=> Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)



    #Training loop — now only runs over neg_loader
    for epoch in range(1, n_epochs+1):
        model.train()
        running_loss = 0.0

        for imgs, _ in tqdm(neg_loader, desc = 'Working on it!'):      # all labels here are 0
            imgs = imgs.to(device)
            optimizer.zero_grad()

            recon = model(imgs)
            loss  = criterion(recon, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(neg_loader.dataset)
        print(f'Epoch {epoch}/{n_epochs} — Neg-only Loss: {epoch_loss:.6f}')

    # switch to eval mode
    model.eval()

    # collect latents for the *normal* class only (or whichever class you want)
    
    latents = []
    for img, label in tqdm(train_ds):    # or use a DataLoader with batch_size
        if label != 0:                   # pick class 0 as “normal”
            continue
        x = img.unsqueeze(0).to(device)
        with torch.no_grad():
            z = model.encoder(x)         # shape (1, latent_dim)
        latents.append(z.cpu().squeeze().numpy())

    latents_normal = np.stack(latents)

    # Extract latents and true labels
    model.eval()
    latents_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            z = model.encoder(imgs)                 # shape (batch, latent_dim)
            latents_list.append(z.cpu().numpy())
            labels_list.append(labels.numpy())

    latents_test = np.concatenate(latents_list, axis=0)      # (N_test, latent_dim)
    true_labels = np.concatenate(labels_list, axis=0)   # (N_test,)


    # Save the arrays
    np.savez(
        output_path,
        latents_normal=latents_normal,
        latents_test=latents_test,
        true_labels=true_labels
    )


if plot:
# (Optional) Subsample for faster t-SNE
    max_samples = 2000
    if len(latents_test) > max_samples:
        idx = np.random.choice(len(latents_test), max_samples, replace=False)
        latents_test = latents_test[idx]
        true_labels  = true_labels[idx]

    # Compute 2D t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42)
    z2 = tsne.fit_transform(latents_test)

    #  Plot
    plt.figure()
    plt.scatter(z2[true_labels == 0, 0], z2[true_labels == 0, 1],
                label='Normal', alpha=0.6)
    plt.scatter(z2[true_labels == 1, 0], z2[true_labels == 1, 1],
                label='Tumor', alpha=0.6)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE of Autoencoder Latent Space')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "tsne_plot.png"))


