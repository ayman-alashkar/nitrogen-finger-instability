# Nitrogen-Water Interface Finger Instability Analysis

A Python-based image analysis pipeline for detecting and quantifying finger instabilities that form when liquid nitrogen is poured over water.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

When liquid nitrogen contacts water, the rapid temperature difference creates vapor-driven instabilities at the interface, forming characteristic "finger" patterns. This project provides tools to:

1. **Detect** the circular container boundary in experimental images
2. **Extract** angular intensity profiles around the interface
3. **Analyze** the periodicity using spectral methods (FFT/PSD)
4. **Quantify** the dominant finger count across varying experimental conditions

## Methodology

The analysis follows these steps:

```
Image → Circle Detection → Annular Sampling → 2D Binning → PSD Averaging → Peak Detection
```

1. **Circle Detection**: Automatic detection of the container boundary using the Hough Transform, and failback to manual detection by clicking at three points on the container rim.
2. **Annular Sampling**: Extract pixels from 60% to 90% of container radius
3. **2D Binning**: Bin pixels by angle (θ) and radius (r) to create intensity grid
4. **PSD Computation**: Compute Power Spectral Density for each radial bin
5. **PSD Averaging**: Average all PSDs to get robust frequency estimate
6. **Peak Detection**: Find dominant frequency = finger count

For detailed methodology, see [docs/methodology.md](docs/methodology.md).

## Installation

```bash
# Clone the repository
git clone https://github.com/ayman-alashkar/nitrogen-finger-instability.git
cd nitrogen-finger-instability

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Single Image Analysis

```bash
python scripts/run_single_image.py --image path/to/image.jpg
```

### Batch Analysis

```bash
python scripts/run_batch_analysis.py --folder path/to/images/
```

### Configuration

Edit `scripts/config.py` to adjust parameters:

```python
CONFIG = {
    "radius_start": 0.6,      # Inner radius (60% of container)
    "radius_end": 0.9,        # Outer radius (90% of container)
    "n_radial_bins": 32,      # Number of radial bins
    "n_angular_bins": 360,    # Angular resolution
    "smooth_sigma": 2,        # Gaussian smoothing
}
```

## Project Structure

```
nitrogen-finger-instability/
│
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
│
├── docs/
│   ├── methodology.md           # Detailed methodology explanation
│   └── figures/                 # Diagrams and illustrations
│
├── src/
│   ├── __init__.py
│   ├── circle_detection.py      # Container boundary detection
│   ├── intensity_analysis.py    # Angular intensity extraction
│   ├── spectral_analysis.py     # PSD and peak detection
│   └── utils.py                 # Helper functions
│
├── scripts/
│   ├── config.py                # Configuration parameters
│   ├── run_single_image.py      # Single image analysis
│   └── run_batch_analysis.py    # Batch processing
│
├── notebooks/
│   └── analysis_walkthrough.ipynb   # Step-by-step tutorial
│
├── data/
│   └── sample/                  # Sample images for testing
│
└── results/
    ├── figures/                 # Output plots
    └── analysis_results.csv     # Results table
```

## Example Results

| Diameter | Temperature | Fingers (5s) | Fingers (10s) |
|----------|-------------|--------------|---------------|
| d1       | T5          | 7            | 7             |
| d1       | T10         | 6            | 6             |
| d1       | T20         | 6            | 6             |
| d1       | T40         | 5            | 5             |
| d2       | T5          | 8            | 8             |
| ...      | ...         | ...          | ...           |

## Experimental Setup

- **Container diameters**: d1, d2, d3, d4
- **Water temperatures**: 5°C, 10°C, 20°C, 40°C
- **Image capture times**: 5s and 10s after nitrogen pour
- **File naming**: `{diameter}_{temperature}_{time}.JPG` (e.g., `d1_T20_1.JPG`)

## Dependencies

- Python ≥ 3.8
- NumPy
- SciPy
- OpenCV
- Matplotlib
- Pandas

## Author

Ayman Alashkar  
Complex Fluids and Flows Unit  
Okinawa Institute of Science and Technology (OIST)

## Supervisor

Prof. Marco Edoardo Rosti

Giulio Foggi Rota

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.
