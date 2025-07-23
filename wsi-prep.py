import openslide
import numpy as np
import cv2
from patchify import patchify
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import os
import glob
import pickle

def get_annotations(annot_path, slide, level):

    """returns list of polygon coordinates of tumor annotations from xml file"""

    tree = ET.parse(annot_path)
    root = tree.getroot()

    downsample = slide.level_downsamples[level]

    polygons = []
    for annot in root.findall('.//Annotation'):
        if annot.get('PartOfGroup') != 'Tumor':
            continue
        coords = []
        for c in annot.findall('.//Coordinate'):
            x = float(c.get('X')) / downsample
            y = float(c.get('Y')) / downsample
            coords.append((x, y))
        if coords:
            polygons.append(coords)
    
    return polygons

def get_tumor_mask(polygons, slide, level):

    """returns tumor mask of slide as np array"""

    level_dims = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    mask = Image.new('L', level_dims, 0)
    draw = ImageDraw.Draw(mask)

    for poly in polygons:
        draw.polygon(poly, outline=255, fill=255)

    mask_array = np.array(mask, dtype=np.uint8)

    return mask_array

def get_tissue_mask(slide, level):

    thumbnail_dims = slide.level_dimensions[level]
    slide_thumbnail  = np.array(slide.read_region([0,0], level, thumbnail_dims))
    slide_hsv = cv2.cvtColor(slide_thumbnail, cv2.COLOR_RGB2HSV)
    val, mask_thumbnail = cv2.threshold(slide_hsv[:,:,1], 0, 255, cv2.THRESH_OTSU)
    mask_thumbnail = cv2.morphologyEx(mask_thumbnail, cv2.MORPH_CLOSE, np.ones((5, 5)))

    return mask_thumbnail

def get_tiles(slide, mask_thumbnail, tumor_mask = None, level = 6, target_level = 4, patch_size = 256, tissue_thresh = 0.2):
        
        
    thumbnail_dims = slide.level_dimensions[level]
    dims_target = slide.level_dimensions[target_level]
    scale = thumbnail_dims[0] / dims_target[0]
    low_mag_patch = int(np.ceil(patch_size * scale))

    low_mag_patches = patchify(mask_thumbnail, (low_mag_patch, low_mag_patch), step = low_mag_patch)

    mask_for_tumor = tumor_mask if tumor_mask is not None else np.zeros_like(mask_thumbnail)
    tumor_mask_patches = patchify(mask_for_tumor, (low_mag_patch, low_mag_patch), step = low_mag_patch)

    tissue_patches = []
    low_mag_coords = []
    total_px = low_mag_patch ** 2

    for i in range(low_mag_patches.shape[0]):
        for j in range(low_mag_patches.shape[1]):
            patch = low_mag_patches[i , j]
            if np.count_nonzero(patch) > tissue_thresh * total_px:
                y = int(i * low_mag_patch / scale)
                x = int(j * low_mag_patch / scale)
                if np.count_nonzero(tumor_mask_patches[i, j]) > 0:
                    tissue_patches.append((y, x, True))
                    low_mag_coords.append((i, j, True))
                else:
                    tissue_patches.append((y, x, False))
                    low_mag_coords.append((i, j, False))

    return tissue_patches, low_mag_coords



def build_slide_index(images_dir, annotations_dir, level, target_level, patch_size, base_mag = 40.0):

    slide_index = {}

    slide_paths = glob.glob(os.path.join(images_dir, '**', '*.tif'),
                        recursive=True)


    for slide_path in slide_paths:
        fname = os.path.basename(slide_path)
        slide_id, _ = os.path.splitext(fname)
        label = slide_id.split('_')[0]

        annot_subdir = os.path.join(annotations_dir, label)
        annot_path = os.path.join(annot_subdir, slide_id + '.xml')
        if not os.path.exists(annot_path):
            annot_path = None

        slide = openslide.OpenSlide(slide_path)
        downsample = slide.level_downsamples[target_level]
        mag_at_level = base_mag / downsample

        mask_thumb = get_tissue_mask(slide, level)
        tumor_mask = None
        if annot_path:
            tumor_mask = get_tumor_mask(
                get_annotations(annot_path, slide, level),
                slide, 
                level)
        
        patches, vis_coords =  get_tiles(
            slide = slide,
            mask_thumbnail = mask_thumb,
            tumor_mask= tumor_mask,
            level = level,
            target_level= target_level,
            patch_size = patch_size
        )

        slide_index[slide_id] = {
            'slide_path': slide_path,
            'label': label,
            'annotation_path': annot_path,
            'level': target_level,
            'work_level': level,
            'visualisation_coords': vis_coords,
            'magnification': mag_at_level,
            'patches': patches,
        }
    
    return slide_index


# index = build_slide_index(
#     images_dir='images',
#     annotations_dir='annotations',
#     level=5,               
#     target_level=3,        
#     patch_size=256,
#     base_mag=40.0          

# )


# # save to disk
# with open('slide_index.pkl', 'wb') as f:
#     pickle.dump(index, f)


# 1) Grab one slide to figure out the lowest‐resolution level
working_level = 7

# 2) Loop through all target levels
for target_level in range(0, 7):
    index = build_slide_index(
        images_dir     = 'images',
        annotations_dir= 'annotations',
        level          = working_level,
        target_level   = target_level,
        patch_size     = 256,
        base_mag       = 40.0
    )

    # 3) Save out
    fname = f'index_level_{target_level}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(index, f)

    print(f"✅ Saved {fname} ({len(index)} slides indexed)")
