"""
Demo for using Custom Phantom with LCD_CT toolkit (Python)
This demo shows how to define custom insert locations and run the LCD analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.lcdct.LCD import measure_LCD, plot_results
from src.lcdct.select_inserts_interactive import select_inserts_interactive

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
    print("Creating synthetic data...")
    sp, sa, config = create_synthetic_data()

    print("Running LCD analysis with programmatic custom configuration...")
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
