# Methodology: Finger Instability Detection

This document provides a detailed explanation of the image analysis pipeline used to detect and quantify finger instabilities at the nitrogen-water interface.

## Table of Contents

1. [Physical Background](#1-physical-background)
2. [Analysis Pipeline Overview](#2-analysis-pipeline-overview)
3. [Step 1: Circle Detection](#3-step-1-circle-detection)
4. [Step 2: Coordinate Transformation](#4-step-2-coordinate-transformation)
5. [Step 3: Annular Sampling and 2D Binning](#5-step-3-annular-sampling-and-2d-binning)
6. [Step 4: Intensity Profile Extraction](#6-step-4-intensity-profile-extraction)
7. [Step 5: High-Pass Filtering](#7-step-5-high-pass-filtering)
8. [Step 6: Power Spectral Density (PSD)](#8-step-6-power-spectral-density-psd)
9. [Step 7: PSD Averaging](#9-step-7-psd-averaging)
10. [Step 8: Peak Detection](#10-step-8-peak-detection)
11. [Parameter Selection](#11-parameter-selection)

---

## 1. Physical Background

When liquid nitrogen is poured onto water, the extreme temperature difference causes rapid vaporization. This creates a vapor layer between the nitrogen and water surfaces, leading to:

- **Interface instabilities** driven by vapor flow
- **Finger patterns** that radiate outward from the contact region

We are studying if the number and morphology of these fingers depend on:
- Container diameter
- Water temperature
- Room temperature

---


Our goal is to **automatically count** these fingers using spectral analysis.

---

## 2. Analysis Pipeline Overview

```
┌─────────────────┐
│   Input Image   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Circle Detection│   ← Hough Transform
└────────┬────────┘   ← Fallback Method
         ▼
┌─────────────────┐
│   Coordinate    │  ← Cartesian to Polar
│  Transformation │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Annular Sampling│  ← 60% to 90% radius
│  + 2D Binning   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Intensity I(θ)  │  ← For each radial bin
│   Extraction    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  High-Pass      │  ← Remove DC leakage (f < 4)
│   Filtering     │
└────────┬────────┘
         ▼
┌─────────────────┐
│  PSD Computation│  ← FFT
└────────┬────────┘
         ▼
┌─────────────────┐
│  PSD Averaging  │  ← Average across radial bins
└────────┬────────┘
         ▼
┌─────────────────┐
│ Peak Detection  │  ← Dominant frequency = finger count
└────────┬────────┘
         ▼
┌─────────────────┐
│  Finger Count   │
└─────────────────┘
```

---

## 3. Step 1: Circle Detection

The circular container boundary is detected using the:

 **Primary Method**: Hough Circle Transform on a grayscale image.
 

### Algorithm

```python
circles = cv2.HoughCircles(
    image_gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=100,
    param2=30,
    minRadius=100,
    maxRadius=400
)
```
**Fallback Method**: Manual selection of 3 points on the rim to compute circle geometry.

### Algorithm


```python
def calculate_circle_from_points(points):
  
    p1, p2, p3 = points
    
    D = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    if D == 0:
        return None, None

    ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / D
    uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / D
    
    center = (int(ux), int(uy))
    radius = int(np.sqrt((p1[0] - ux)**2 + (p1[1] - uy)**2))
    
    return center, radius
```

### Output
- **Center**: (cx, cy) in pixel coordinates
- **Radius**: r in pixels

### Notes
- Images are rescaled to 30%
- Results are scaled back to original resolution
- For T=40°C images (high vapor), the manual method is required due to the unclear rim because of the dense fog.

---

## 4. Step 2: Coordinate Transformation

We transform from Cartesian (x, y) to polar (r, θ) coordinates centered on the container.

### Equations

For each pixel at position (x, y):

$$r = \sqrt{(x - c_x)^2 + (y - c_y)^2}$$

$$\theta = \arctan2(y - c_y, x - c_x)$$

Where:
- $(c_x, c_y)$ is the detected circle center
- $\theta$ is converted to degrees [0°, 360°)

---

## 5. Step 3: Annular Sampling and 2D Binning

We analyze an **annular region** from 60% to 90% of the container radius. This region captures the finger pattern while avoiding:
- The central nitrogen pool (< 60%)
- Edge effects near the container wall (> 90%)

### 2D Binning

All pixels in the annulus are binned by both θ and r:

```python
intensity_grid, _, _, _ = stats.binned_statistic_2d(
    theta_values,      # Angular positions
    r_values,          # Radial positions
    intensity_values,  # Pixel intensities
    statistic='mean',
    bins=[theta_bins, r_bins]  # [360, 32] bins
)
```

### Output
- **Intensity grid**: Shape (360, 32)
  - 360 angular bins (1° resolution)
  - 32 radial bins (from 60% to 90% radius)

---

## 6. Step 4: Intensity Profile Extraction

For each radial bin $i$, we extract the angular intensity profile:

$$I_i(\theta) = \text{intensity\_grid}[:, i]$$

This gives us 32 independent intensity profiles, one for each radial distance.

---

## 7. Step 5: High-Pass Filtering


We apply a high-pass filter to remove frequencies below f = 4:

```python
def highpass_filter(signal, cutoff_freq=3):
    nyquist = len(signal) / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normalized_cutoff, btype='high')
    filtered = filtfilt(b, a, signal)
    return filtered
```

---

## 8. Step 6: Power Spectral Density (PSD)

The **Power Spectral Density** reveals which frequencies (= finger counts) are present in the signal.

### Computation

1. Remove mean and linear trend:
   $$y(\theta) = I(\theta) - \bar{I} - \text{trend}$$

2. Apply FFT:
   $$\hat{Y}(f) = \mathcal{F}\{y(\theta)\}$$

3. Compute power:
   $$PSD(f) = \frac{1}{N} |\hat{Y}(f)|^2$$

### Frequency Interpretation

The frequency axis represents **cycles per 360°**, which directly corresponds to **finger count**:

- f = 5 → 5 complete oscillations around the circle → 5 fingers
- f = 8 → 8 complete oscillations around the circle → 8 fingers

---

## 9. Step 7: PSD Averaging

Instead of analyzing a single radial strip, we **average PSDs across all radial bins**.

### Why Average PSDs?

Each radial bin gives an independent estimate of the finger count. By averaging:

- **Noise reduction**: Random fluctuations cancel out
- **Robustness**: One noisy radial bin doesn't skew the result
- **Phase independence**: Even if fingers shift slightly between radii, the frequency content is preserved

### Implementation

```python
all_psds = []
for i in range(n_radial_bins):
    intensities = intensity_grid[:, i]
    freq, psd = compute_psd(intensities)
    all_psds.append(psd)

avg_psd = np.mean(all_psds, axis=0)
```

### Visualization

```
Radius 60%:  PSD₁  ──┐
Radius 61%:  PSD₂  ──┤
Radius 62%:  PSD₃  ──┼──→  Average  ──→  Dominant Peak
    ...       ...    │
Radius 90%:  PSD₃₂ ──┘
```

---

## 10. Step 8: Peak Detection

The **dominant peak** in the averaged PSD corresponds to the finger count.

### Algorithm

1. Restrict search to physical range: f ∈ [4, 15]
   - f < 4: Too few fingers (likely noise or artifacts)
   - f > 15: Too many fingers (likely noise)

2. Find maximum power in this range:
   ```python
   mask = (freq >= 4) & (freq <= 15)
   peak_freq = freq[mask][np.argmax(psd[mask])]
   ```

### Output

The peak frequency directly gives the finger count:
- Peak at f = 6 → **6 fingers**

---

## 11. Parameter Selection

### Recommended Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `radius_start` | 0.6 | Avoid central nitrogen pool |
| `radius_end` | 0.9 | Avoid container edge effects |
| `n_radial_bins` | 32 | Balance between resolution and noise |
| `n_angular_bins` | 360 | 1° resolution |
| `smooth_sigma` | 2 | Reduce high-frequency noise |
| `highpass_cutoff` | 4 | Remove DC leakage |
| `freq_min` | 4 | Minimum physical finger count |
| `freq_max` | 15 | Maximum physical finger count |

### Adjusting Parameters

- **More fingers expected?** Increase `freq_max`
- **Noisy images?** Increase `smooth_sigma` or `n_radial_bins`
- **Fingers near edge?** Increase `radius_end` toward 0.95
- **Large central pool?** Increase `radius_start` toward 0.7

---
