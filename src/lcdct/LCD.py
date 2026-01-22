"""
Low Contrast Detectability (LCD)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mode
from typing import Union, List, Optional, Any, Tuple

from .utils import (
    get_demo_truth_masks, 
    get_insert_radius, 
    get_roi_from_truth_mask,
    read_mhd
)
from .Observers import LG_CHO, DOG_CHO, Gabor_CHO, NPWE

def get_roi_from_manual_selection(img: np.ndarray, x_center: int, y_center: int, r: int) -> np.ndarray:
    """Extract a square region from the image centered on (x_center, y_center) with radius r (half-width).

    Args:
        img: (N, Y, X) image stack.
        x_center: Center x coordinate.
        y_center: Center y coordinate.
        r: Radius (half-width) of the crop.

    Returns:
        np.ndarray: Cropped image stack (N, 2*r, 2*r).
    """
    rows, cols = img.shape[1], img.shape[2]

    x_min = max(0, x_center - r)
    x_max = min(cols, x_center + r)
    y_min = max(0, y_center - r)
    y_max = min(rows, y_center + r)

    # Ensure square output (pad if near border? or just clip?)
    # MATLAB get_ROI_from_manual_selection just slices.
    # If near border, size might be smaller.

    return img[:, y_min:y_max, x_min:x_max]


def measure_LCD(signal_present: np.ndarray, signal_absent: np.ndarray, ground_truth: Union[np.ndarray, str, Path, List[dict]],
                observers: Optional[List[Union[str, Any]]] = None, n_reader: int = 10, pct_split: float = 0.5, seed_split: Optional[Union[List[int], np.ndarray]] = None) -> pd.DataFrame:
    """Calculates Low Contrast Detectability (LCD) metrics (AUC, SNR).

    Args:
        signal_present: np.ndarray (N, Y, X) of signal present images.
        signal_absent: np.ndarray (N, Y, X) of signal absent images.
        ground_truth: np.ndarray (Y, X) ground truth image OR Path to mhd file OR List of dicts (custom config).
                      Custom config format: [{'x': int, 'y': int, 'r': float, 'HU': float}, ...]
        observers: List of strings (e.g., 'LG_CHO_2D') or Observer instances. Default: ['LG_CHO_2D'].
        n_reader: Number of readers (bootstraps/splits).
        pct_split: Train/test split ratio (0.0 to 1.0).
        seed_split: List/array of seeds or None.

    Returns:
        pd.DataFrame: DataFrame containing detailed results for each insert and observer.
    """
    if observers is None:
        observers = ['LG_CHO_2D']

    use_custom_config = False
    phantom_config = []

    # Handle ground truth if it is a path (string/Path)
    if isinstance(ground_truth, (str, Path)):
        ground_truth = read_mhd(str(ground_truth))
    elif isinstance(ground_truth, list):
        use_custom_config = True
        phantom_config = ground_truth

    # Process inputs
    # Ensure (N, Y, X)
    if signal_present.ndim != 3:
        raise ValueError("signal_present must be 3D (N, Y, X)")
    if signal_absent.ndim != 3:
        raise ValueError("signal_absent must be 3D (N, Y, X)")
        
    results_list = []
    
    if use_custom_config:
        n_inserts = len(phantom_config)
        valid_indices = range(n_inserts)
        # Determine crop sizes
        # legacy behavior: use max diameter for all crops?
        # MATLAB: crop_r = max(insert_rs).
        # Then uses current_insert_r for manual crop? No, in measure_LCD refactor:
        # sp_imgs = get_ROI_from_manual_selection(..., current_insert_r)
        # So it uses individual radius.
        
    else:
        # Get truth masks
        # truth_masks: (Y, X, N_inserts)
        truth_masks = get_demo_truth_masks(ground_truth)
        n_inserts = truth_masks.shape[2]

        # Determine crop sizes
        insert_rs = []
        valid_indices = []
        for i in range(n_inserts):
            mask = truth_masks[:, :, i]
            if np.sum(mask) < 1:
                continue
            valid_indices.append(i)
            r = get_insert_radius(mask)
            insert_rs.append(r)

        if not valid_indices:
            print("No valid inserts found in ground truth.")
            return pd.DataFrame()

        crop_r = max(insert_rs) # max diameter
    
    # Loop observers
    for obs_item in observers:
        # Loop inserts
        for idx in valid_indices:
            
            if use_custom_config:
                cfg = phantom_config[idx]
                x_center = int(round(cfg['x']))
                y_center = int(round(cfg['y']))
                current_insert_r = float(cfg['r']) # this is radius
                # Note: MATLAB code treats input 'r' as radius, but get_insert_radius returns Diameter?
                # measure_LCD.m: insert_diameter_pix = 2*insert_r
                # get_insert_radius in MATLAB: returns diameter?
                # Python utils.py get_insert_radius: returns diameter (bbox max dim).
                # Python measure_LCD (old): insert_r = get_insert_radius -> Diameter.
                # LG_CHO channel width = 2/3 * insert_r (Diameter).
                # crop_r = max(Diameter).
                # get_roi_from_truth_mask(..., nx=2*Diameter). -> nx is half width? No, nx argument in get_roi...
                # Let's align carefully.
                # If custom config 'r' is RADIUS, then Diameter = 2*r.
                # If 'r' is Diameter...
                # The user will provide Radius usually (click center + radius).
                # MATLAB select_inserts_interactive returns x, y.
                # measure_LCD1 passed insert_r (scalar).
                # Our new MATLAB code expects phantom_config.r (Radius).
                # measure_LCD.m refactored:
                # insert_diameter_pix = 2*current_insert_r.
                # So current_insert_r is RADIUS.

                # So here, let's assume cfg['r'] is RADIUS.
                radius_pix = int(round(current_insert_r))
                diameter_pix = 2 * current_insert_r

                # Extract ROIs
                # get_ROI_from_manual_selection takes radius
                sp_rois = get_roi_from_manual_selection(signal_present, x_center, y_center, radius_pix)
                sa_rois = get_roi_from_manual_selection(signal_absent, x_center, y_center, radius_pix)

                insert_hu_val = cfg.get('HU', 0)

            else:
                truth_mask = truth_masks[:, :, idx]
                diameter_pix = get_insert_radius(truth_mask) # Diameter

                # ROI extraction
                # Legacy: nx = 2 * crop_r (Max Diameter)
                sp_rois = get_roi_from_truth_mask(truth_mask, signal_present, nx=2*crop_r)
                sa_rois = get_roi_from_truth_mask(truth_mask, signal_absent, nx=2*crop_r)

                # Determine Insert HU
                vals = ground_truth[truth_mask > 0]
                if len(vals) > 0:
                    insert_hu_val = mode(vals, axis=None).mode
                    if isinstance(insert_hu_val, np.ndarray):
                         insert_hu_val = insert_hu_val[0]
                else:
                    insert_hu_val = 0

            # Check ROI validity
            if sp_rois is None or sa_rois is None or sp_rois.size == 0:
                continue
                
            # Create/Configure Observer
            obs_name = obs_item if isinstance(obs_item, str) else obs_item.type

            # Instantiate observer
            current_obs = None
            if isinstance(obs_item, str):
                name = obs_item.upper()
                if name == 'LG_CHO_2D':
                    current_obs = LG_CHO(sp_rois, sa_rois, channel_width=2/3 * diameter_pix)
                elif name == 'DOG_CHO_2D':
                    current_obs = DOG_CHO(sp_rois, sa_rois)
                elif name == 'GABOR_CHO_2D':
                    current_obs = Gabor_CHO(sp_rois, sa_rois)
                elif name == 'NPWE_2D':
                    current_obs = NPWE(sp_rois, sa_rois)
                else:
                    raise ValueError(f"Unknown observer: {name}")
            else:
                current_obs = obs_item
                # If LG_CHO, manually update if possible
                if hasattr(current_obs, 'channel_width'):
                     current_obs.channel_width = 2/3 * diameter_pix
                # Update signals
                current_obs.signal_present = sp_rois - sp_rois.mean(axis=(1, 2), keepdims=True)
                current_obs.signal_absent = sa_rois - sa_rois.mean(axis=(1, 2), keepdims=True)
            
            # Run Study
            df_res = current_obs.run_study(n_readers=n_reader, pct_split=pct_split, seed=seed_split)
            
            # Append metadata
            df_res['insert_HU'] = insert_hu_val
            df_res['insert_diameter_pix'] = diameter_pix
            df_res['insert_idx'] = idx
            
            # Add to list
            results_list.append(df_res)
            
    if not results_list:
        return pd.DataFrame()
        
    final_df = pd.concat(results_list, ignore_index=True)
    return final_df

import matplotlib.pyplot as plt

def plot_results(results: pd.DataFrame, ylim: Optional[Tuple[float, float]] = None) -> None:
    """Plots Low Contrast Detectability (LCD) results (AUC).

    Args:
        results: DataFrame containing LCD results (must contain 'auc', 'insert_HU', 'observer').
        ylim: Tuple of (min, max) for y-axis limits.
    """
    if results.empty:
        print("No results to plot.")
        return
        
    dose_levels = np.sort(results['dose_level'].unique()) if 'dose_level' in results else [None]
    insert_hus = np.sort(results['insert_HU'].unique())
    observers = results['observer'].unique()
    
    if 'recon' not in results.columns:
         results['recon'] = 'Unknown'
    recons = results['recon'].unique()
    
    n_inserts = len(insert_hus)
    
    # Grid size
    if n_inserts == 1:
        nrows, ncols = 1, 1
    elif n_inserts == 2:
        nrows, ncols = 1, 2
    else:
        nrows, ncols = 2, 2 # Max 4 usually
        
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, hu in enumerate(insert_hus):
        ax = axes[idx]
        subset = results[results['insert_HU'] == hu]
        
        # Group by observer, recon
        groups = subset.groupby(['observer', 'recon'])
        
        for name, group in groups:
            obs_name, recon_name = name
            # If multiple doses
            if len(dose_levels) > 1 and dose_levels[0] is not None:
                # Plot AUC vs Dose
                # Aggregate across readers
                agg = group.groupby('dose_level')['auc'].agg(['mean', 'std']).reset_index()
                ax.errorbar(agg['dose_level'], agg['mean'], yerr=agg['std'], label=f"{recon_name} {obs_name}", capsize=3)
                ax.set_xlabel('Dose Level')
            else:
                # Bar plot for single dose?
                # or just plot scatter
                mean_auc = group['auc'].mean()
                std_auc = group['auc'].std()
                ax.bar(f"{recon_name}\n{obs_name}", mean_auc, yerr=std_auc, capsize=5)
        
        ax.set_title(f"Insert HU: {hu}")
        ax.set_ylabel("AUC")
        if ylim:
            ax.set_ylim(ylim)
        ax.legend()
        
    plt.tight_layout()
    plt.show()    

