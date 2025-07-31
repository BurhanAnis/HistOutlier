#!/usr/bin/env python3
"""
Script to extract 256x256 patches from an existing slide-index pickle and save them either
as PNG files organized by label (normal vs. tumor), or pack them into an LMDB or HDF5 database
for fast deep-learning data loading.

Dependencies:
    pip install openslide-python pillow numpy lmdb h5py

Usage:
    python save_patches.py \
        --index_file index_level_4.pkl \
        --patch_size 256 \
        --out_dir image_patches \
        --backend lmdb \
        --lmdb_path patches_lmdb \
        # or --backend fs (file system), or --backend hdf5 --hdf5_path patches.h5
"""
import os
import io
import argparse
import pickle
import numpy as np
from PIL import Image
import openslide

try:
    import lmdb
except ImportError:
    lmdb = None

try:
    import h5py
except ImportError:
    h5py = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and store slide patches for deep learning.")
    parser.add_argument(
        '--index_file', type=str, required=True,
        help="Path to pickle file containing the slide_index dict.")
    parser.add_argument(
        '--patch_size', type=int, default=256,
        help="Size of patches (in pixels) at the target level.")
    parser.add_argument(
        '--out_dir', type=str, default='image_patches',
        help="Output base directory for file-system backend (image_patches/normal and tumor).")
    parser.add_argument(
        '--backend', type=str, default='fs', choices=['fs', 'lmdb', 'hdf5'],
        help="Storage backend: 'fs' (filesystem), 'lmdb', or 'hdf5'.")
    parser.add_argument(
        '--lmdb_path', type=str, default='patches_lmdb',
        help="Path to LMDB environment directory (if backend=lmdb).")
    parser.add_argument(
        '--hdf5_path', type=str, default='patches.h5',
        help="Path to HDF5 file (if backend=hdf5).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the slide index
    with open(args.index_file, 'rb') as f:
        slide_index = pickle.load(f)

    # Prepare backends
    if args.backend == 'fs':
        for lbl in ['normal', 'tumor']:
            os.makedirs(os.path.join(args.out_dir, lbl), exist_ok=True)
    elif args.backend == 'lmdb':
        if lmdb is None:
            raise ImportError("lmdb package not installed. Install with 'pip install lmdb'.")
        env = lmdb.open(args.lmdb_path, map_size=1 << 40)
        txn = env.begin(write=True)
    elif args.backend == 'hdf5':
        if h5py is None:
            raise ImportError("h5py package not installed. Install with 'pip install h5py'.")
        # Estimate total patches for pre-allocation
        total_patches = sum(len(info['patches']) for info in slide_index.values())
        h5f = h5py.File(args.hdf5_path, 'w')
        # Store images as uint8 arrays (H, W, C)
        dset_img = h5f.create_dataset(
            'images', shape=(total_patches, args.patch_size, args.patch_size, 3),
            dtype='uint8', chunks=True)
        dset_lbl = h5f.create_dataset(
            'labels', shape=(total_patches,), dtype='uint8')
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    count = 0
    # Iterate slides and extract patches
    for slide_id, info in slide_index.items():
        slide_path = info['slide_path']
        print(f"Processing {slide_id} ({len(info['patches'])} patches)")
        slide = openslide.OpenSlide(slide_path)
        level = info['level']
        downsample = slide.level_downsamples[level]

        for (y, x, is_tumor) in info['patches']:
            # Convert level-coordinates to level-0
            loc = (int(x * downsample), int(y * downsample))
            patch = slide.read_region(loc, level, (args.patch_size, args.patch_size)).convert('RGB')

            if args.backend == 'fs':
                subdir = 'tumor' if is_tumor else 'normal'
                filename = f"{slide_id}_{count:06d}.png"
                patch.save(os.path.join(args.out_dir, subdir, filename))

            elif args.backend == 'lmdb':
                key = f"{slide_id}_{count:06d}".encode('ascii')
                buffer = io.BytesIO()
                patch.save(buffer, format='PNG')
                txn.put(key, buffer.getvalue())

            elif args.backend == 'hdf5':
                arr = np.array(patch, dtype='uint8')
                dset_img[count, ...] = arr
                dset_lbl[count] = 1 if is_tumor else 0

            count += 1

        slide.close()

    # Finalize backends
    if args.backend == 'lmdb':
        txn.commit()
        env.close()
    elif args.backend == 'hdf5':
        h5f.close()

    print(f"Done! Extracted and stored {count} patches.")

if __name__ == '__main__':
    main()
