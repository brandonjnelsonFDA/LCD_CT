---
title: LCD_CT
app_file: gradio_app.py
sdk: gradio
sdk_version: 6.3.0
---

# Low Contrast Detectability (LCD) for CT Toolbox

The **LCD-CT Toolbox** provides a comprehensive interface to evaluate the low contrast detectability (LCD) performance of CT image reconstruction and denoising algorithms. It uses Model Observers (MO) to calculate the AUC of targets in the MITA-LCD phantom.

## 🚀 Ways to Use LCD-CT

### 1. Web Application (Hugging Face Spaces)
The easiest way to use the tool without installation is via our live web application on Hugging Face:

[**launch LCD-CT Web App**](https://huggingface.co/spaces/bnel1201/LCD_CT)

**Features:**
- **Upload**: Support for DICOM and MHD file uploads.
- **Visuals**: Interactive viewer with stack averaging and window/level controls.
- **Analysis**: Automated ROI selection and LCD measurement.

### 2. Local GUI Application
Run the web interface locally on your machine for faster processing and local file access.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App:**
   ```bash
   python gradio_app.py
   ```
   The app will open in your browser at `http://localhost:7861`.

### 3. Programmatic Usage
For researchers and developers, the toolbox offers Python and MATLAB/Octave APIs for batch processing and integration into pipelines.
See `README.rst` or the `demos/` folder for detailed scripts.

## 🔗 Complementary Tools
For additional resources on measuring Low Contrast Detectability, we recommend checking out **CTpro.net**, which offers helpful tools and educational material for CT image quality assessment.

- [**CTpro.net**](https://ctpro.net)

---
*For detailed citations and regulatory information, please refer to [README.rst](README.rst).*
