import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Union, List, Optional
from pathlib import Path

def select_inserts_interactive(image_volume: np.ndarray, n_inserts: int, output_filename: Optional[Union[str, Path]] = None) -> List[dict]:
    """
    Interactively select insert centers from an image.

    Args:
        image_volume: 3D or 2D image array. If 3D, the center slice is used for display.
        n_inserts: Number of inserts to select.
        output_filename: (Optional) Path to save the coordinates (json).

    Returns:
        List[dict]: List of dicts with 'x' and 'y' keys.
    """

    # Handle 3D volume by taking center slice
    if image_volume.ndim == 3:
        slice_idx = image_volume.shape[0] // 2
        img_display = image_volume[slice_idx, :, :]
    elif image_volume.ndim == 2:
        img_display = image_volume
    else:
        raise ValueError("Input image must be 2D or 3D")

    # Setup figure
    fig, ax = plt.subplots()
    ax.imshow(img_display, cmap='gray')
    ax.axis('off')
    ax.set_title(f"Select {n_inserts} insert locations (Left click to select)")

    print(f"Please select {n_inserts} insert locations on the figure window...")

    # Get inputs
    # ginput returns [(x1, y1), (x2, y2), ...]
    # timeout=-1 means wait indefinitely
    pts = plt.ginput(n=n_inserts, timeout=-1, show_clicks=True)

    plt.close(fig)

    coords = [{'x': p[0], 'y': p[1]} for p in pts]

    if output_filename:
        with open(output_filename, 'w') as f:
            json.dump(coords, f, indent=2)
        print(f"Coordinates saved to {output_filename}")

    return coords
