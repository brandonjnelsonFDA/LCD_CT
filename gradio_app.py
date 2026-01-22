
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw

from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import hough_circle, hough_circle_peaks

# Add src to path robustly
current_dir = Path(__file__).resolve().parent
src_path = current_dir / 'src'

# Check if src is already in path to avoid duplicates
if str(src_path) not in sys.path and src_path.exists():
    sys.path.insert(0, str(src_path))

# Also ensure current dir is in path for local modules if needed
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from lcdct.LCD import measure_LCD
except ImportError:
    # Final fallback if direct import fails
    try:
        sys.path.append(str(src_path)) 
        from lcdct.LCD import measure_LCD
    except ImportError as e:
        print(f"CRITICAL: Could not import lcdct.LCD. Path: {sys.path}. Error: {e}")
        measure_LCD = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

# Global or Session State helper functions
def generate_synthetic_data(sz=(10, 100, 100)):
    background = np.zeros(sz) + 100
    inserts = np.zeros(sz)
    centers = [(30, 50), (70, 50)] # (x, y)
    radius = 10
    insert_hus = [110, 120]
    
    y_grid, x_grid = np.ogrid[:sz[1], :sz[2]]
    
    for idx, (cx, cy) in enumerate(centers):
        mask = ((x_grid - cx)**2 + (y_grid - cy)**2) <= radius**2
        for z in range(sz[0]):
            inserts[z][mask] = insert_hus[idx] - 100
    
    sp_data = background + inserts + np.random.randn(*sz) * 5
    sa_data = background + np.random.randn(*sz) * 5
    return sp_data, sa_data

def load_data_from_files(files):
    if not files:
        return None
    if sitk is None:
        raise ImportError("SimpleITK not installed.")
        
    try:
        # Sort files by name to ensure correct sequence
        # files is a list of file paths (temp paths from Gradio)
        files = sorted(files)
        
        # Check extensions
        exts = [os.path.splitext(f)[1].lower() for f in files]
        
        if '.mhd' in exts:
            # Assume mhd/raw pair. Load the mhd
            mhd_file = [f for f in files if f.endswith('.mhd')][0]
            img = sitk.ReadImage(mhd_file)
            arr = sitk.GetArrayFromImage(img)
            return arr
            
        elif '.npy' in exts:
            # Single NPY file
            arr = np.load(files[0])
            return arr
            
        else:
            # Assume DICOM Series or image stack
            # sitk.ImageSeriesReader expects filenames.
            # Gradio provides full paths.
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(files)
            img = reader.Execute()
            arr = sitk.GetArrayFromImage(img)
            return arr
            
    except Exception as e:
        print(f"Error loading files: {e}")
        return None

def draw_overlays(img_arr, centers, radius, color=(255, 0, 0)):
    """
    img_arr: 2D numpy array (normalized 0-255 uint8)
    centers: list of dict {'x', 'y', 'r'}
    """
    # Convert to RGB for colored overlays
    if img_arr.ndim == 2:
        img_rgb = np.stack([img_arr]*3, axis=-1)
    else:
        img_rgb = img_arr.copy()
        
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    for c in centers:
        cx, cy = c['x'], c['y']
        r = c.get('r', radius)
        # Draw crosshair
        draw.line([(cx - 5, cy), (cx + 5, cy)], fill=color, width=1)
        draw.line([(cx, cy - 5), (cx, cy + 5)], fill=color, width=1)
        # Draw circle
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=color, width=1)
        
    return np.array(pil_img)

def normalize_for_display(img, vmin, vmax):
    clipped = np.clip(img, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin)
    return (norm * 255).astype(np.uint8)

def create_app():
    with gr.Blocks(title="LCD CT Analysis Tool", theme=gr.themes.Default(spacing_size="sm")) as demo:
        gr.Markdown("# LCD CT Analysis Tool")
        
        # State variables
        sp_state = gr.State(None) # (N, Y, X)
        sa_state = gr.State(None) # (N, Y, X)
        centers_state = gr.State([]) # List of dicts
        slice_idx_state = gr.State(0) 
        view_mode_state = gr.State("slice") # "slice" or "average"
        
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Data & Config")
                
                with gr.Tab("Synthetic"):
                    load_synth_btn = gr.Button("Load Synthetic Data")
                
                with gr.Tab("Upload"):
                    sp_files = gr.File(label="Signal Present (DICOM/MHD/NPY)", file_count="multiple")
                    sa_files = gr.File(label="Signal Absent (DICOM/MHD/NPY)", file_count="multiple")
                    load_upload_btn = gr.Button("Load Uploaded Data")
                
                gr.Markdown("---")
                # Config
                radius_input = gr.Number(value=10, label="Insert Radius (px)")
                hu_input = gr.Number(value=120, label="Insert HU")
                auto_size_btn = gr.Button("Auto-size Radius & Centers")
                
                # Window/Level
                with gr.Row():
                    window_input = gr.Number(value=80, label="Window")
                    level_input = gr.Number(value=0, label="Level")
                
                clear_btn = gr.Button("Clear Centers")
                analyze_btn = gr.Button("Run Analysis", variant="primary")
                status_box = gr.Textbox(label="Status", value="Ready", interactive=False)
                
            with gr.Column(scale=4):
                gr.Markdown("### Viewer")
                with gr.Row(equal_height=True):
                    slider = gr.Slider(minimum=0, maximum=1, step=1, label="Slice Index", value=0, scale=4)
                    toggle_view_btn = gr.Button("View Stack Average", variant="secondary", scale=1)
                
                with gr.Row():
                    # Increased height for better viewing
                    sp_image = gr.Image(label="Signal Present (Click to Select)", interactive=True, type="numpy", height=600)
                    sa_image = gr.Image(label="Signal Absent", interactive=False, type="numpy", height=600)
        
        with gr.Row():
            results_df = gr.Dataframe(label="Results", headers=["insert_idx", "insert_HU", "auc"])
            results_plot = gr.Plot(label="AUC Summary")
            
        download_btn = gr.File(label="Download CSV")

        # --- Functions ---
        
        def load_synthetic():
            sp, sa = generate_synthetic_data()
            msg = "Synthetic Data Loaded"
            return (
                sp, 
                sa, 
                [], # centers
                sp.shape[0] // 2, # slice
                gr.update(maximum=sp.shape[0]-1, value=sp.shape[0]//2), # slider
                msg
            )
            
        def load_uploaded(sp_files_list, sa_files_list):
            if not sp_files_list:
                return None, None, [], 0, gr.update(), "No SP files provided."
            
            try:
                # Load SP
                sp = load_data_from_files(sp_files_list)
                if sp is None:
                    return None, None, [], 0, gr.update(), "Failed to load SP files."
                if sp.ndim == 2: sp = sp[np.newaxis, ...]
                
                # Load SA
                sa = None
                if sa_files_list:
                    sa = load_data_from_files(sa_files_list)
                    if sa is not None and sa.ndim == 2: sa = sa[np.newaxis, ...]
                
                msg = "Loaded SP." + (" Loaded SA." if sa is not None else " Warning: No SA loaded.")
                
                return (
                    sp,
                    sa,
                    [],
                    sp.shape[0] // 2,
                    gr.update(maximum=sp.shape[0]-1, value=sp.shape[0]//2),
                    msg
                )
            except Exception as e:
                return None, None, [], 0, gr.update(), f"Error: {e}"

        def toggle_view_mode(current_mode):
            if current_mode == "slice":
                return "average", "View Slices"
            else:
                return "slice", "View Stack Average"

        def update_view(sp, sa, idx, centers, window, level, radius, view_mode):
            if sp is None:
                return None, None
            
            # Use Window/Level to calc vmin/vmax
            try:
                w = float(window)
                l = float(level)
            except:
                w = 400
                l = 40
            
            vmin = l - w / 2
            vmax = l + w / 2
            
            # Determine image content based on view_mode
            if view_mode == "average":
                img_sp = np.mean(sp, axis=0)
                if sa is not None:
                     img_sa = np.mean(sa, axis=0)
                else:
                     img_sa = np.zeros_like(img_sp)
            else:
                # Clamp index
                idx = max(0, min(int(idx), sp.shape[0]-1))
                img_sp = sp[idx]
                img_sa = sa[idx] if sa is not None else np.zeros_like(img_sp)
            
            # Normalize
            disp_sp = normalize_for_display(img_sp, vmin, vmax)
            disp_sa = normalize_for_display(img_sa, vmin, vmax)
            
            # Overlays on SP
            disp_sp_marked = draw_overlays(disp_sp, centers, radius, color=(255, 0, 0))
            # Overlays on SA
            disp_sa_marked = draw_overlays(disp_sa, centers, radius, color=(0, 0, 255))
            
            return disp_sp_marked, disp_sa_marked

        def on_click(evt: gr.SelectData, centers, radius):
            # Append new center
            new_center = {'x': evt.index[0], 'y': evt.index[1], 'r': radius}
            # Avoid dupes logic could go here
            updated_centers = centers + [new_center]
            return updated_centers

        def clear_centers():
            return []

        def auto_size(sp, centers, current_radius):
            if sp is None:
                return centers, current_radius, "No data loaded."
            
            # Mean intensity projection (Already used for calculation logic, regardless of view)
            img_avg = np.mean(sp, axis=0)
            img_norm = (img_avg - img_avg.min()) / (img_avg.max() - img_avg.min())
            
            new_centers = []
            updates_count = 0
            
            if not centers:
                # Global search
                edges = canny(img_norm, sigma=3)
                hough_radii = np.arange(5, 30, 1)
                hough_res = hough_circle(edges, hough_radii)
                accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
                
                if len(radii) > 0:
                    r = int(radii[0])
                    # Assuming cx, cy are x, y coordinates
                    new_centers.append({'x': cx[0], 'y': cy[0], 'r': r})
                    return new_centers, r, f"Found global circle at ({cx[0]}, {cy[0]}) r={r}"
                else:
                    return centers, current_radius, "Global search failed."
            
            # Local refinement
            for c in centers:
                cx, cy = c['x'], c['y']
                cr = c.get('r', current_radius)
                
                # Search margin
                search_margin = int(max(cr * 2, 20))
                rows, cols = img_avg.shape
                
                y_min = max(0, int(cy) - search_margin)
                y_max = min(rows, int(cy) + search_margin)
                x_min = max(0, int(cx) - search_margin)
                x_max = min(cols, int(cx) + search_margin)
                
                roi = img_norm[y_min:y_max, x_min:x_max]
                
                if roi.size == 0:
                    new_centers.append(c)
                    continue
                    
                try:
                    thresh = threshold_otsu(roi)
                    mask = roi > thresh
                    lbl = label(mask)
                    regions = regionprops(lbl)
                    
                    if not regions:
                        new_centers.append(c)
                        continue
                        
                    roi_cy, roi_cx = roi.shape[0] // 2, roi.shape[1] // 2
                    best_region = None
                    min_dist = float('inf')
                    
                    for r_prop in regions:
                        rc_y, rc_x = r_prop.centroid
                        dist = (rc_x - roi_cx)**2 + (rc_y - roi_cy)**2
                        if dist < min_dist:
                            min_dist = dist
                            best_region = r_prop
                            
                    if best_region:
                        l_cy, l_cx = best_region.centroid
                        best_cx = l_cx + x_min
                        best_cy = l_cy + y_min
                        best_r = best_region.equivalent_diameter_area / 2
                        
                        new_centers.append({'x': best_cx, 'y': best_cy, 'r': best_r})
                        updates_count += 1
                    else:
                        new_centers.append(c)
                except Exception as e:
                    print(f"Error refining {c}: {e}")
                    new_centers.append(c)

            # Update radius to mean of new centers
            if new_centers:
                avg_r = np.mean([c['r'] for c in new_centers])
                return new_centers, int(round(avg_r)), f"Refined {updates_count} centers."
            
            return centers, current_radius, "No updates."

        def run_analysis(sp, sa, centers, hu, radius):
            if sp is None or sa is None:
                return None, None, None
            if not centers:
                return None, None, None
            
            config = []
            for c in centers:
                config.append({
                    'x': c['x'],
                    'y': c['y'],
                    'r': c['r'],
                    'HU': hu
                })
            
            observers = ['LG_CHO_2D']
            try:
                res = measure_LCD(sp, sa, config, observers)
                
                # Plot
                fig, ax = plt.subplots()
                if not res.empty:
                    res.groupby('insert_idx')['auc'].mean().plot(kind='bar', ax=ax)
                    ax.set_title("AUC by Insert Index")
                    ax.set_ylabel("AUC")
                    plt.tight_layout()
                
                # Save CSV
                csv_path = "results.csv"
                res.to_csv(csv_path, index=False)
                
                return res, fig, csv_path
            except Exception as e:
                print(f"Error: {e}")
                return pd.DataFrame({'Error': [str(e)]}), None, None

        # --- Wiring ---
        
        def load_defaults():
            base_path = current_dir / 'data' / 'small_dataset' / 'fbp' / 'dose_100'
            sp_path = base_path / 'signal_present' / 'signal_present.mhd'
            sa_path = base_path / 'signal_absent' / 'signal_absent.mhd'
            
            sp, sa = None, None
            msg = "Default data not found."
            
            if sp_path.exists():
                try:
                    img_sp = sitk.ReadImage(str(sp_path))
                    sp = sitk.GetArrayFromImage(img_sp)
                except Exception as e:
                    print(f"Failed to load default SP: {e}")
            
            if sa_path.exists():
                try:
                    img_sa = sitk.ReadImage(str(sa_path))
                    sa = sitk.GetArrayFromImage(img_sa)
                except Exception as e:
                     print(f"Failed to load default SA: {e}")
                     
            if sp is not None:
                msg = "Default Data Loaded"
                return (
                    sp, 
                    sa, 
                    [], # centers
                    sp.shape[0] // 2, # slice
                    gr.update(maximum=sp.shape[0]-1, value=sp.shape[0]//2), # slider
                    msg
                )
            else:
                 # Fallback to no data or synthetic if requested, but for now just empty
                 return (None, None, [], 0, gr.update(), "Default data not found.")

        # --- Wiring ---
        
        # Load Defaults on Launch
        demo.load(
            load_defaults,
            inputs=[],
            outputs=[sp_state, sa_state, centers_state, slice_idx_state, slider, status_box]
        ).then(
            update_view,
            inputs=[sp_state, sa_state, slice_idx_state, centers_state, window_input, level_input, radius_input, view_mode_state],
            outputs=[sp_image, sa_image]
        )

        # Load Synthetic
        load_synth_btn.click(
            load_synthetic, 
            inputs=[], 
            outputs=[sp_state, sa_state, centers_state, slice_idx_state, slider, status_box]
        ).then(
            update_view,
            inputs=[sp_state, sa_state, slice_idx_state, centers_state, window_input, level_input, radius_input, view_mode_state],
            outputs=[sp_image, sa_image]
        )
        
        # Load Uploaded
        load_upload_btn.click(
            load_uploaded,
            inputs=[sp_files, sa_files],
            outputs=[sp_state, sa_state, centers_state, slice_idx_state, slider, status_box]
        ).then(
            update_view,
            inputs=[sp_state, sa_state, slice_idx_state, centers_state, window_input, level_input, radius_input, view_mode_state],
            outputs=[sp_image, sa_image]
        )
        
        # Toggle View Mode
        toggle_view_btn.click(
            toggle_view_mode,
            inputs=[view_mode_state],
            outputs=[view_mode_state, toggle_view_btn]
        ).then(
             update_view,
            inputs=[sp_state, sa_state, slice_idx_state, centers_state, window_input, level_input, radius_input, view_mode_state],
            outputs=[sp_image, sa_image]
        )
        
        # Slice Change
        slider.change(
            lambda i: i, inputs=[slider], outputs=[slice_idx_state]
        ).then(
            update_view,
            inputs=[sp_state, sa_state, slice_idx_state, centers_state, window_input, level_input, radius_input, view_mode_state],
            outputs=[sp_image, sa_image]
        )
        
        # Display settings change
        window_input.change(update_view, inputs=[sp_state, sa_state, slice_idx_state, centers_state, window_input, level_input, radius_input, view_mode_state], outputs=[sp_image, sa_image])
        level_input.change(update_view, inputs=[sp_state, sa_state, slice_idx_state, centers_state, window_input, level_input, radius_input, view_mode_state], outputs=[sp_image, sa_image])
        
        # Auto-Size
        auto_size_btn.click(
            auto_size,
            inputs=[sp_state, centers_state, radius_input],
            outputs=[centers_state, radius_input, status_box]
        ).then(
            update_view,
            inputs=[sp_state, sa_state, slice_idx_state, centers_state, window_input, level_input, radius_input, view_mode_state],
            outputs=[sp_image, sa_image]
        )
        
        # Click
        sp_image.select(
            on_click,
            inputs=[centers_state, radius_input],
            outputs=[centers_state]
        ).then(
            update_view,
            inputs=[sp_state, sa_state, slice_idx_state, centers_state, window_input, level_input, radius_input, view_mode_state],
            outputs=[sp_image, sa_image]
        )
        
        # Clear
        clear_btn.click(clear_centers, outputs=[centers_state]).then(
            update_view,
            inputs=[sp_state, sa_state, slice_idx_state, centers_state, window_input, level_input, radius_input, view_mode_state],
            outputs=[sp_image, sa_image]
        )
        
        # Analyze
        analyze_btn.click(
            run_analysis,
            inputs=[sp_state, sa_state, centers_state, hu_input, radius_input],
            outputs=[results_df, results_plot, download_btn]
        )

    return demo

if __name__ == "__main__":
    app = create_app()
    
    # Check if running on Hugging Face Spaces
    if "SPACE_ID" in os.environ:
        # Hugging Face handles port (7860) and SSL automatically
        app.launch()
    else:
        # Local execution: specifying port 7861 to avoid conflicts
        app.launch(server_name="0.0.0.0", server_port=7861, share=True)
