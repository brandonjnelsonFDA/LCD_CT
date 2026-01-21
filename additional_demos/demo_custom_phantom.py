"""
Demo for using Custom Phantom with LCD_CT toolkit (Python)
This demo shows how to define custom insert locations and run the LCD analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Ensure src is in path if running from repo
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))
    # print(f"Added {src_path} to sys.path") 

from lcdct.LCD import measure_LCD, plot_results
from lcdct.select_inserts_interactive import select_inserts_interactive

def create_synthetic_data(sz=(10, 100, 100)):
    background = np.zeros(sz) + 100
    inserts = np.zeros(sz)

    # Define inserts
    centers = [(30, 50), (70, 50)] # (x, y)
    radius = 10
    insert_hus = [110, 120]

    ground_truth = background.copy()

    y_grid, x_grid = np.ogrid[:sz[1], :sz[2]]

    config = []

    for idx, (cx, cy) in enumerate(centers):
        mask = ((x_grid - cx)**2 + (y_grid - cy)**2) <= radius**2

        # Apply to all slices
        for z in range(sz[0]):
            inserts[z][mask] = insert_hus[idx] - 100
            ground_truth[z][mask] = insert_hus[idx]

        config.append({
            'x': cx,
            'y': cy,
            'r': radius,
            'HU': insert_hus[idx]
        })

    signal_present = background + inserts + np.random.randn(*sz) * 5
    signal_absent = background + np.random.randn(*sz) * 5

    return signal_present, signal_absent, config

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Custom Phantom LCD Demo')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode to select inserts')
    args = parser.parse_args()

    print("Creating synthetic data...")
    sp, sa, initial_config = create_synthetic_data()

    if args.interactive:
        print("Running in INTERACTIVE mode...")
        print("Please select 2 insert locations in the window.")
        # We need the radius and HU from the initial config or defined constants
        # In this simple demo, we'll assume same radius/HU for all selected points 
        # as defined in create_synthetic_data: radius=10, HU=[110, 120]
        
        # Note: select_inserts_interactive returns list of dicts [{'x':.., 'y':..}, ..]
        # We use the signal_present image for selection
        selected_coords = select_inserts_interactive(sp, n_inserts=2)
        
        config = []
        # We'll assign the HUs and radius to the selected coordinates in order
        # Default values from create_synthetic_data
        default_hus = [110, 120]
        radius = 10
        
        for idx, coord in enumerate(selected_coords):
            # If user selects more than we have HUs for, just cycle or use last? 
            # The demo asks for 2 inserts, so we should be fine.
            hu = default_hus[idx] if idx < len(default_hus) else default_hus[-1]
            
            config.append({
                'x': coord['x'],
                'y': coord['y'],
                'r': radius,
                'HU': hu
            })
            
        print("Interactive config selection complete.")
        
    else:
        print("Running LCD analysis with programmatic custom configuration...")
        config = initial_config

    print("Configuration to be used:")
    for c in config:
        print(c)

    observers = ['LG_CHO_2D']

    # Pass config as ground_truth argument
    results = measure_LCD(sp, sa, config, observers)

    print("Results:")
    print(results)

    plot_results(results)
    plt.savefig('docs/source/custom_phantom_results.png')
    print("Saved plot to docs/source/custom_phantom_results.png")

if __name__ == "__main__":
    main()
