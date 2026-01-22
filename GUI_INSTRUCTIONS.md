# LCD-CT GUI Application

This is a graphical user interface for the LCD-CT Toolbox, allowing users to interactively select insert centers and run Low Contrast Detectability analysis.

## Requirements

- Python 3.8+
- Dependencies listed in `pyproject.toml` (installed via `pip install -e .` or `conda env create -f environment.yml`)
- `tkinter` (usually included with Python)

## How to Run

1. Activate your LCD_CT environment:
   ```bash
   conda activate LCD_CT
   ```

2. Run the application:
   ```bash
   python lcd_gui.py
   ```

## Usage

1. **Start Up**: The app launches with synthetic demo data loaded by default.
2. **Load Data**: Use "Load Signal Present Series" and "Load Signal Absent Series" to load your own DICOM or MHD image series.
3. **View & Select**:
   - Scroll with the mouse wheel to navigate through slices.
   - Click "View Stack Average" to see the mean projection (helps in finding inserts).
   - Left-click on the center of the inserts to mark them. Red circles will appear.
   - Adjust "Insert Radius" in the settings panel if needed.
4. **Analyze**: Click "Confirm Centers & Analyze" to run the LCD measurement.
5. **Results**:
   - Results (AUC) will be displayed in the results panel.
   - A plot will show the performance for each insert HU.
   - Click "Download Results (CSV)" to save the full metrics to a file.
