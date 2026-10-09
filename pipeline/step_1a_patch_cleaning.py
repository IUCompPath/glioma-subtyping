import os
import argparse
import h5py
import numpy as np
import pandas as pd
import openslide
import cv2
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool, cpu_count
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity

def filter_patch(patch_array, patch_mag, intensity_thresh=225, intensity_thresh_b=50):
    """
    Comprehensive filtering logic based on magnification-specific thresholds.
    """
    patch_size = patch_array.shape[0]
    total_pixels = patch_size * patch_size

    # --- Pre-calculations ---
    # White pixel percentage (Intensity)
    count_white_pixels = np.where(np.all(patch_array > intensity_thresh, axis=-1))[0]
    percent_pixels = len(count_white_pixels) / total_pixels

    # Black pixel percentage (Intensity)
    intensity_thresh_b_low = 128
    count_black_pixels = np.where(np.all(patch_array < intensity_thresh_b_low, axis=-1))[0]
    percent_pixel_b = len(count_black_pixels) / total_pixels

    # HSV conversion
    patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_RGB2HSV)
    h_channel = patch_hsv[:, :, 0] # Hue
    s_channel = patch_hsv[:, :, 1] # Saturation
    v_channel = patch_hsv[:, :, 2] # Value

    # Calculated metrics
    percent_s_low = len(np.where(s_channel < intensity_thresh_b)[0]) / total_pixels
    percent_v_high = len(np.where(v_channel > intensity_thresh)[0]) / total_pixels
    percent_h_low = len(np.where(h_channel < 128)[0]) / total_pixels
    
    # HED Stain Detection
    ihc_hed = rgb2hed(patch_array)
    e_channel = rescale_intensity(ihc_hed[:, :, 1], out_range=(0, 255), 
                                 in_range=(0, np.percentile(ihc_hed[:, :, 1], 99)))
    percent_e_low = len(e_channel[e_channel < 50]) / total_pixels

    # --- Initial Universal Filter (Empty/No Stain) ---
    if percent_e_low > 0.85 or percent_h_low > 0.9:
        return False

    # --- Magnification Specific Logic ---
    if patch_mag == '2.5x':
        if percent_s_low > 0.98 or np.mean(s_channel) < 5 or percent_v_high > 0.98:
            if not percent_s_low < 0.25: return False
        elif (percent_s_low > 0.95 and percent_v_high > 0.95) or percent_pixel_b > 0.95 or percent_pixels > 0.75:
            return False

    elif patch_mag == '5x':
        if percent_s_low > 0.98 or np.mean(s_channel) < 5 or percent_v_high > 0.98:
            if not percent_s_low < 0.25: return False
        elif (percent_s_low > 0.9 and percent_v_high > 0.9) or percent_pixel_b > 0.95 or percent_pixels > 0.7:
            return False

    elif patch_mag == '10x':
        if percent_s_low > 0.975 or np.mean(s_channel) < 5 or percent_v_high > 0.975:
            if not percent_s_low < 0.25: return False
        elif (percent_s_low > 0.88 and percent_v_high > 0.88) or percent_pixel_b > 0.9 or percent_pixels > 0.6:
            return False

    elif patch_mag == '20x' or patch_mag == '40x':
        if percent_s_low > 0.9 or np.mean(s_channel) < 5 or percent_v_high > 0.9:
            if not percent_s_low < 0.25: return False
        elif (percent_s_low > 0.85 and percent_v_high > 0.9) or percent_pixel_b > 0.9 or percent_pixels > 0.6:
            return False

    return True

def process_slide_cleanup(slide_id, args):
    slide_name = os.path.splitext(slide_id)[0]
    # Locate H5 file
    h5_files = glob(os.path.join(args.h5_dir, f"{slide_name}*"))
    if not h5_files: return f"Skip: {slide_name} (H5 not found)"
    h5_path = h5_files[0]
    
    # Locate WSI file (check multiple common extensions)
    wsi_path = None
    for ext in ['.svs', '.ndpi', '.tif']:
        path_check = os.path.join(args.wsi_dir, slide_name + ext)
        if os.path.exists(path_check):
            wsi_path = path_check
            break
    if not wsi_path: return f"Skip: {slide_name} (WSI not found)"

    try:
        slide = openslide.OpenSlide(wsi_path)
        with h5py.File(h5_path, "r") as f:
            coords = f['coords'][()]
            attrs = dict(f['coords'].attrs)
        
        patch_size = attrs['patch_size']
        patch_level = attrs['patch_level']
        
        # Determine valid coordinates
        mask = []
        for coord in coords:
            patch = np.array(slide.read_region(coord, patch_level, (patch_size, patch_size)).convert("RGB"))
            mask.append(filter_patch(patch, args.patching))
        
        valid_coords = coords[np.array(mask)]
        
        # Overwrite file with new coordinates
        with h5py.File(h5_path, "w") as f:
            dset = f.create_dataset('coords', data=valid_coords)
            for k, v in attrs.items(): dset.attrs[k] = v
            
        return f"Cleaned {slide_name}: {len(coords)} -> {len(valid_coords)}"
    except Exception as e:
        return f"Error {slide_name}: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', type=str, required=True)
    parser.add_argument('--h5_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--patching', type=str, required=True, 
                        choices=['2.5x', '5x', '10x', '20x', '40x'])
    parser.add_argument('--num_workers', type=int, default=cpu_count()-2)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    slides = df['slide_id'].tolist()

    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(
            lambda s: process_slide_cleanup(s, args), slides), total=len(slides)))
    
    # Write summary to file
    with open("cleanup_log.txt", "w") as f:
        f.write("\n".join(results))