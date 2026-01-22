import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.draw import disk
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).resolve().parent
src_path = current_dir / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from lcdct.LCD import measure_LCD
    from lcdct.utils import read_mhd, get_demo_truth_masks, get_insert_radius
except ImportError:
    # Fallback for development if src is not found or structure differs
    print("Warning: Could not import lcdct.LCD. Ensure src is in python path.")

class LCDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LCD CT Analysis Tool")
        self.root.geometry("1400x900")

        # Data State
        self.sp_data = None # Signal Present (N, Y, X)
        self.sa_data = None # Signal Absent (N, Y, X)
        self.current_slice_idx = 0
        self.selected_centers = [] # List of (x, y)
        self.config = [] # List of dicts
        self.view_mode = "slice" # "slice" or "average"
        self.results = None

        # Defaults
        self.default_radius = 10
        self.default_hu = 120
        self.default_window = 400
        self.default_level = 40

        # UI Layout
        self._create_top_bar()
        self._create_main_area()
        self._create_control_panel()
        
        # Load Default Data
        self.load_default_synthetic_data()

    def _create_top_bar(self):
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top_frame, text="Load Signal Present Series", command=self.load_sp_series).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Load Signal Absent Series", command=self.load_sa_series).pack(side=tk.LEFT, padx=5)
        
        tk.Frame(top_frame, width=20).pack(side=tk.LEFT) # Spacer

        tk.Button(top_frame, text="Confirm Centers & Analyze", command=self.confirm_and_analyze, bg="#ddffdd").pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Download Results (CSV)", command=self.download_results).pack(side=tk.LEFT, padx=5)

    def _create_main_area(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: Controls/Info
        left_panel = tk.Frame(main_frame, width=200, bg="#f0f0f0")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        tk.Label(left_panel, text="Settings", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=10)
        
        tk.Label(left_panel, text="Insert Radius (px):", bg="#f0f0f0").pack(anchor="w")
        self.radius_var = tk.IntVar(value=self.default_radius)
        tk.Entry(left_panel, textvariable=self.radius_var).pack(fill=tk.X, padx=5)

        tk.Label(left_panel, text="Insert HU:", bg="#f0f0f0").pack(anchor="w", pady=5)
        self.hu_var = tk.IntVar(value=self.default_hu)
        tk.Entry(left_panel, textvariable=self.hu_var).pack(fill=tk.X, padx=5)

        tk.Label(left_panel, text="Status:", font=("Arial", 10, "bold"), bg="#f0f0f0", pady=20).pack(anchor="w")
        self.status_label = tk.Label(left_panel, text="Ready", bg="#f0f0f0", wraplength=180, justify="left")
        self.status_label.pack(anchor="w", fill=tk.X)

        tk.Label(left_panel, text="Display Settings", font=("Arial", 12, "bold"), bg="#f0f0f0", pady=10).pack(anchor="w")
        
        tk.Label(left_panel, text="Window:", bg="#f0f0f0").pack(anchor="w")
        self.window_var = tk.IntVar(value=self.default_window)
        w_entry = tk.Entry(left_panel, textvariable=self.window_var)
        w_entry.pack(fill=tk.X, padx=5)
        w_entry.bind('<Return>', self.apply_window_level)
        w_entry.bind('<FocusOut>', self.apply_window_level)

        tk.Label(left_panel, text="Level:", bg="#f0f0f0").pack(anchor="w")
        self.level_var = tk.IntVar(value=self.default_level)
        l_entry = tk.Entry(left_panel, textvariable=self.level_var)
        l_entry.pack(fill=tk.X, padx=5)
        l_entry.bind('<Return>', self.apply_window_level)
        l_entry.bind('<FocusOut>', self.apply_window_level)

        tk.Button(left_panel, text="Brain (80/40)", command=lambda: self.set_preset(80, 40)).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(left_panel, text="Liver (300/100)", command=lambda: self.set_preset(300, 100)).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(left_panel, text="Soft Tissue (400/40)", command=lambda: self.set_preset(400, 40)).pack(fill=tk.X, padx=5, pady=2)

        # Center: Image Viewer
        self.center_panel = tk.Frame(main_frame, bg="black")
        self.center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.ax_sp = self.fig.add_subplot(121)
        self.ax_sa = self.fig.add_subplot(122)
        self.ax_sp.axis('off')
        self.ax_sa.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.center_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Event bindings
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Right: Results
        self.right_panel = tk.Frame(main_frame, width=400, bg="white")
        # Right: Results
        self.right_panel = tk.Frame(main_frame, width=400, bg="white")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        tk.Label(self.right_panel, text="Results & ROI Manager", font=("Arial", 12, "bold"), bg="white").pack(pady=10)
        
        # ROI Table
        columns = ("id", "r", "x", "y", "mean", "std", "auc")
        self.tree = ttk.Treeview(self.right_panel, columns=columns, show="headings", height=8)
        self.tree.heading("id", text="#")
        self.tree.column("id", width=30)
        self.tree.heading("r", text="R")
        self.tree.column("r", width=40)
        self.tree.heading("x", text="X")
        self.tree.column("x", width=50)
        self.tree.heading("y", text="Y")
        self.tree.column("y", width=50)
        self.tree.heading("mean", text="MeanHU")
        self.tree.column("mean", width=60)
        self.tree.heading("std", text="StdHU")
        self.tree.column("std", width=60)
        self.tree.heading("auc", text="AUC")
        self.tree.column("auc", width=60)
        self.tree.pack(fill=tk.X, padx=5, pady=5)

        self.results_text = tk.Text(self.right_panel, height=5, width=40)
        self.results_text.pack(fill=tk.X, padx=5)
        
        # Placeholder for Mini Plot
        self.res_fig = Figure(figsize=(4, 4), dpi=100)
        self.res_ax = self.res_fig.add_subplot(111)
        self.res_canvas = FigureCanvasTkAgg(self.res_fig, master=self.right_panel)
        self.res_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def _create_control_panel(self):
        control_frame = tk.Frame(self.root, pady=5)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.view_stack_btn = tk.Button(control_frame, text="View Stack Average", command=self.toggle_stack_average)
        self.view_stack_btn.pack(side=tk.LEFT, padx=20)
        
        tk.Button(control_frame, text="Auto-size Radius", command=self.auto_size_radius).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Clear Selected Centers", command=self.clear_selection).pack(side=tk.LEFT, padx=20)

        self.slice_label = tk.Label(control_frame, text="Slice: 0/0")
        self.slice_label.pack(side=tk.RIGHT, padx=20)

    def load_default_synthetic_data(self):
        self.update_status("Generatng synthetic data...")
        # Recreate the logic from demo_custom_phantom.py
        sz = (10, 100, 100)
        background = np.zeros(sz) + 100
        inserts = np.zeros(sz)
        centers = [(30, 50), (70, 50)] # (x, y)
        radius = 10
        insert_hus = [110, 120]
        
        y_grid, x_grid = np.ogrid[:sz[1], :sz[2]]
        
        # Generate inserts
        for idx, (cx, cy) in enumerate(centers):
            mask = ((x_grid - cx)**2 + (y_grid - cy)**2) <= radius**2
            for z in range(sz[0]):
                inserts[z][mask] = insert_hus[idx] - 100
        
        self.sp_data = background + inserts + np.random.randn(*sz) * 5
        self.sa_data = background + np.random.randn(*sz) * 5
        
        self.update_status("Loaded synthetic data.")
        self.current_slice_idx = sz[0] // 2
        self.refresh_view()

    def update_status(self, msg):
        self.status_label.config(text=msg)
        self.root.update_idletasks()

    def load_sp_series(self):
        path = filedialog.askdirectory(title="Select Signal Present Directory")
        if not path: return
        self.update_status(f"Loading SP from {path}...")
        try:
             # Basic implementation: sitk read or folder iteration
             # Using a simplified version of utils.load_dataset logic for just one folder
             import SimpleITK as sitk
             try:
                 # Check for mhd
                 mhd_files = list(Path(path).glob("*.mhd"))
                 if mhd_files:
                     img = sitk.ReadImage(str(mhd_files[0]))
                     arr = sitk.GetArrayFromImage(img)
                 else:
                     # Load series
                     reader = sitk.ImageSeriesReader()
                     dicom_names = reader.GetGDCMSeriesFileNames(path)
                     reader.SetFileNames(dicom_names)
                     img = reader.Execute()
                     arr = sitk.GetArrayFromImage(img)
                 
                 self.sp_data = arr
                 self.update_status("SP Loaded. Load SA next.")
                 if self.sp_data.ndim == 2:
                     self.sp_data = self.sp_data[np.newaxis, ...]
                 self.current_slice_idx = self.sp_data.shape[0] // 2
                 self.refresh_view()
                 
             except Exception as e:
                 messagebox.showerror("Error", f"Failed to load images: {e}")
                 
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_sa_series(self):
        path = filedialog.askdirectory(title="Select Signal Absent Directory")
        if not path: return
        self.update_status(f"Loading SA from {path}...")
        try:
             import SimpleITK as sitk
             try:
                 mhd_files = list(Path(path).glob("*.mhd"))
                 if mhd_files:
                     img = sitk.ReadImage(str(mhd_files[0]))
                     arr = sitk.GetArrayFromImage(img)
                 else:
                     reader = sitk.ImageSeriesReader()
                     dicom_names = reader.GetGDCMSeriesFileNames(path)
                     reader.SetFileNames(dicom_names)
                     img = reader.Execute()
                     arr = sitk.GetArrayFromImage(img)
                 
                 self.sa_data = arr
                 if self.sa_data.ndim == 2:
                     self.sa_data = self.sa_data[np.newaxis, ...]
                 self.update_status("SA Loaded.")
             except Exception as e:
                 messagebox.showerror("Error", f"Failed to load images: {e}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def toggle_stack_average(self):
        if self.view_mode == "slice":
            self.view_mode = "average"
            self.view_stack_btn.config(text="View Slices")
        else:
            self.view_mode = "slice"
            self.view_stack_btn.config(text="View Stack Average")
        self.refresh_view()

    def refresh_view(self):
        if self.sp_data is None: return
        
        self.ax_sp.clear()
        self.ax_sa.clear()
        
        if self.view_mode == "slice":
            # Clamp index
            self.current_slice_idx = max(0, min(self.current_slice_idx, self.sp_data.shape[0] - 1))
            img_sp = self.sp_data[self.current_slice_idx]
            if self.sa_data is not None:
                 # Check if SA has same depth, otherwise clamp independently or assume same geometry
                 # For safety, ensure index is valid for SA
                 sa_idx = min(self.current_slice_idx, self.sa_data.shape[0] - 1)
                 img_sa = self.sa_data[sa_idx]
            else:
                 img_sa = np.zeros_like(img_sp)
                 
            self.slice_label.config(text=f"Slice: {self.current_slice_idx + 1}/{self.sp_data.shape[0]}")
        else:
            # AIP
            img_sp = np.mean(self.sp_data, axis=0)
            if self.sa_data is not None:
                img_sa = np.mean(self.sa_data, axis=0)
            else:
                img_sa = np.zeros_like(img_sp)
            self.slice_label.config(text="Stack Average")

        # Apply Window/Level
        window = self.window_var.get()
        level = self.level_var.get()
        vmin = level - window / 2
        vmax = level + window / 2

        self.ax_sp.imshow(img_sp, cmap='gray', vmin=vmin, vmax=vmax)
        self.ax_sp.set_title("Signal Present")
        self.ax_sp.axis('off')

        self.ax_sa.imshow(img_sa, cmap='gray', vmin=vmin, vmax=vmax)
        self.ax_sa.set_title("Signal Absent")
        self.ax_sa.axis('off')

        # Draw selected centers on SP (Red)
        for val in self.selected_centers:
            # Handle backward compatibility if tuple is (x, y)
            if len(val) == 3:
                cx, cy, cr = val
            else:
                cx, cy = val
                cr = self.radius_var.get()
                
            self.ax_sp.plot(cx, cy, 'r+', markersize=10)
            circle = plt.Circle((cx, cy), cr, color='r', fill=False)
            self.ax_sp.add_patch(circle)
            
        # Draw selected centers on SA (Blue)
        for val in self.selected_centers:
            if len(val) == 3:
                cx, cy, cr = val
            else:
                cx, cy = val
                cr = self.radius_var.get()
                
            self.ax_sa.plot(cx, cy, 'b+', markersize=10)
            circle = plt.Circle((cx, cy), cr, color='b', fill=False)
            self.ax_sa.add_patch(circle)
            
        self.canvas.draw()

    def set_preset(self, window, level):
        self.window_var.set(window)
        self.level_var.set(level)
        self.refresh_view()

    def apply_window_level(self, event=None):
        self.refresh_view()

    def on_scroll(self, event):
        if self.view_mode != "slice" or self.sp_data is None: return
        if event.button == 'up':
            self.current_slice_idx = (self.current_slice_idx + 1) % self.sp_data.shape[0]
        elif event.button == 'down':
            self.current_slice_idx = (self.current_slice_idx - 1) % self.sp_data.shape[0]
        self.refresh_view()

    def on_click(self, event):
        if event.inaxes not in [self.ax_sp, self.ax_sa]: return
        if event.button == 1: # Left click
            # Add point
            x, y = event.xdata, event.ydata
            r = self.radius_var.get()
            self.selected_centers.append((x, y, r))
            self.update_status(f"Selected point at ({x:.1f}, {y:.1f})")
            
            self.update_roi_table()
            self.refresh_view()
    
    def update_roi_table(self):
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for idx, val in enumerate(self.selected_centers):
            if len(val) == 3:
                x, y, r = val
            else:
                x, y = val
                r = self.radius_var.get()
                
            mean_val, std_val = self.calculate_roi_stats(x, y, r)
            self.tree.insert("", "end", values=(idx+1, int(r), int(x), int(y), f"{mean_val:.1f}", f"{std_val:.1f}", "-"))

    def calculate_roi_stats(self, x, y, r):
        # Average Mean and Std across slices
        if self.sp_data is None: return 0.0, 0.0
        
        # Create mask
        # We can just compute once on the stack average? 
        # Requirement: "calculated for each slice and then averaged"
        
        roi_means = []
        roi_stds = []
        
        sz = self.sp_data.shape
        y_grid, x_grid = np.ogrid[:sz[1], :sz[2]]
        mask = ((x_grid - x)**2 + (y_grid - y)**2) <= r**2
        
        if np.sum(mask) == 0: return 0.0, 0.0
        
        # Optimize: Apply to whole stack at once
        # sp_data is (N, Y, X)
        # We want pixels inside mask for each slice
        
        # Extract columns corresponding to mask
        # masked_data shape: (N, num_pixels_in_mask)
        masked_data = self.sp_data[:, mask] 
        
        # Mean per slice
        slice_means = np.mean(masked_data, axis=1)
        slice_stds = np.std(masked_data, axis=1)
        
        return np.mean(slice_means), np.mean(slice_stds)

    def auto_size_radius(self):
        if self.sp_data is None:
             messagebox.showerror("Error", "No image data loaded.")
             return
             
        self.update_status("Auto-sizing radius and centers...")
        
        # Stack Average
        img_avg = np.mean(self.sp_data, axis=0)
        img_norm = (img_avg - img_avg.min()) / (img_avg.max() - img_avg.min())
        
        # If no points selected, use global search
        if not self.selected_centers:
            edges = canny(img_norm, sigma=3)
            hough_radii = np.arange(5, 30, 1)
            hough_res = hough_circle(edges, hough_radii)
            accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
            
            if len(radii) > 0:
                found_r = radii[0]
                self.radius_var.set(int(found_r))
                self.update_status(f"Auto-sized radius to {found_r} px (Global).")
            else:
                messagebox.showerror("Error", "Could not detect any circles globally.")
                self.update_status("Auto-size failed.")
        else:
            # Refine each selected center
            new_centers = []
            updates_count = 0
            
            for val in self.selected_centers:
                if len(val) == 3:
                    cx, cy, cr = val
                else:
                    cx, cy = val
                    cr = self.radius_var.get()
                    
                # Define ROI search window (e.g., +/- 2*R)
                search_margin = int(max(cr * 2, 20))
                rows, cols = img_avg.shape
                
                y_min = max(0, int(cy) - search_margin)
                y_max = min(rows, int(cy) + search_margin)
                x_min = max(0, int(cx) - search_margin)
                x_max = min(cols, int(cx) + search_margin)
                
                
                roi = img_norm[y_min:y_max, x_min:x_max]
                
                if roi.size == 0:
                    new_centers.append(val)
                    continue
                
                if roi.size == 0:
                    new_centers.append(val)
                    continue
                
                try:
                    # Intensity-based segmentation
                    # 1. Threshold
                    thresh = threshold_otsu(roi)
                    mask = roi > thresh
                    
                    # 2. Label regions
                    lbl = label(mask)
                    regions = regionprops(lbl)
                    
                    if not regions:
                        new_centers.append(val)
                        continue
                        
                    # 3. Find region closest to original center (which is center of ROI)
                    roi_cy, roi_cx = roi.shape[0] // 2, roi.shape[1] // 2
                    best_region = None
                    min_dist = float('inf')
                    
                    for r_prop in regions:
                        # Centroid is (row, col) -> (y, x)
                        rc_y, rc_x = r_prop.centroid
                        dist = (rc_x - roi_cx)**2 + (rc_y - roi_cy)**2
                        if dist < min_dist:
                            min_dist = dist
                            best_region = r_prop
                    
                    if best_region:
                        # Update center and radius
                        # region centroid is relative to ROI
                        l_cy, l_cx = best_region.centroid
                        best_cx = l_cx + x_min
                        best_cy = l_cy + y_min
                        best_r = best_region.equivalent_diameter_area / 2 
                        
                        new_centers.append((best_cx, best_cy, best_r))
                        updates_count += 1
                    else:
                        new_centers.append(val)
                        
                except Exception as e:
                    print(f"Auto-size failed for point {val}: {e}")
                    new_centers.append(val)
            
            self.selected_centers = new_centers
            
            # Update UI fields with stats from the last found center (or average)
            if new_centers:
                 # Update Radius 
                 avg_r = int(round(np.mean([c[2] for c in new_centers])))
                 self.radius_var.set(avg_r)
                 
                 # Estimate HU
                 cx, cy, cr = new_centers[0]
                 est_mean, _ = self.calculate_roi_stats(cx, cy, cr)
                 self.hu_var.set(int(round(est_mean)))
            
            self.update_roi_table()
            self.refresh_view()
            if updates_count > 0:
                 self.update_status(f"Refined {updates_count} center(s). Radius set to ~{self.radius_var.get()} px.")
            else:
                 self.update_status("No better centers found locally.")

    def clear_selection(self):
        self.selected_centers = []
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.refresh_view()

    def confirm_and_analyze(self):
        if self.sp_data is None or self.sa_data is None:
            messagebox.showerror("Error", "Please load both Signal Present and Absent data.")
            return
        if not self.selected_centers:
            messagebox.showwarning("Warning", "No centers selected. Please select at least one insert center.")
            return
            
        self.update_status("Running LCD Analysis...")
        
        # Build config
        self.config = []
        # Support per-insert radius if available
        # But global HU unless needed
        hu = self.hu_var.get()
        
        for val in self.selected_centers:
            if len(val) == 3:
                x, y, r = val
            else:
                x, y = val
                r = self.radius_var.get()
                
            self.config.append({
                'x': x,
                'y': y,
                'r': r,
                'HU': hu 
            })
            
        try:
            observers = ['LG_CHO_2D']
            self.results = measure_LCD(self.sp_data, self.sa_data, self.config, observers)
            self.update_status("Analysis Complete.")
            self.show_results()
            
            # Update Table AUC
            # Use insert_idx to map results to the correct row
            # Update Table AUC
            # Use insert_idx to map results to the correct row
            
            # Identify the observer name in the results
            # 'LG_CHO_2D' in input maps to 'LG_CHO' class name in output
            target_obs = None
            unique_observers = self.results['observer'].unique()
            
            # Simple heuristic: if we requested LG_CHO_2D, look for LG_CHO or LG_CHO_2D
            if 'LG_CHO_2D' in observers:
                if 'LG_CHO' in unique_observers:
                    target_obs = 'LG_CHO'
                elif 'LG_CHO_2D' in unique_observers:
                    target_obs = 'LG_CHO_2D'
            
            # If still None and we have results, just pick the first one?
            if target_obs is None and len(unique_observers) > 0:
                target_obs = unique_observers[0]

            if target_obs:
                # Clear all first
                for item in self.tree.get_children():
                    self.tree.set(item, "auc", "-")
                    
                obs_res = self.results[self.results['observer'] == target_obs]
                
                for item in self.tree.get_children():
                    # Get ID from tree (1-based index stored in column 0)
                    # Tree values returns a tuple of strings usually
                    vals = self.tree.item(item, "values")
                    row_id = int(vals[0]) - 1 # convert to 0-based insert index to match config/LCD
                    
                    # Find matching result
                    match = obs_res[obs_res['insert_idx'] == row_id]
                    
                    if not match.empty:
                        # Take the mean if multiple rows
                        val = match['auc'].mean() 
                        self.tree.set(item, "auc", f"{val:.3f}")
                    else:
                        self.tree.set(item, "auc", "N/A")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_status(f"Error: {e}")
            messagebox.showerror("Error", f"Analysis failed: {e}")

    def show_results(self):
        if self.results is None or self.results.empty:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No valid results.")
            return
            
        # Text summary
        summary = self.results.groupby(['observer', 'insert_HU'])['auc'].mean().to_string()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, summary)
        
        # Plot
        self.res_ax.clear()
        # Simplified plot: Bar chart of AUC per HU/Observer
        
        # Group by observer and HU
        grouped = self.results.groupby(['observer', 'insert_HU'])['auc'].mean().reset_index()
        
        # Bar chart
        xs = range(len(grouped))
        labels = [f"HU{row['insert_HU']}\n{row['observer']}" for _, row in grouped.iterrows()]
        self.res_ax.bar(xs, grouped['auc'])
        self.res_ax.set_xticks(xs)
        self.res_ax.set_xticklabels(labels, rotation=45, ha='right')
        self.res_ax.set_ylabel("AUC")
        self.res_ax.set_title("LCD Results")
        self.res_fig.tight_layout()
        self.res_canvas.draw()
        
        # Pop up detailed window if needed, but this is a start

    def download_results(self):
        if self.results is None: return
        f = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if f:
            self.results.to_csv(f, index=False)
            self.update_status(f"Saved results to {f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LCDApp(root)
    root.mainloop()
