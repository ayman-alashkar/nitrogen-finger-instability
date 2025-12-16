"""
Single Image Analysis Script

Analyze a single image with detailed visualization.

Usage:
    python run_single_image.py --image path/to/image.jpg
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.circle_detection import detect_circle_from_path
from src.intensity_analysis import extract_intensity_grid, get_annulus_info
from src.spectral_analysis import compute_psd, analyze_averaged_psd, get_all_peaks
from src.utils import load_image, preprocess_image
from scripts.config import CONFIG


def plot_analysis(
    image: np.ndarray,
    center: tuple,
    radius: int,
    intensity_grid: np.ndarray,
    avg_psd: np.ndarray,
    freq: np.ndarray,
    peak_freq: float,
    config: dict,
    save_path: str = None
):
    """
    Create comprehensive visualization of the analysis.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Original image with detected circle and annular region
    ax1 = fig.add_subplot(2, 3, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image_rgb)
    
    # Draw annular region
    r_start = int(radius * config.get("radius_start", 0.6))
    r_end = int(radius * config.get("radius_end", 0.9))
    circle_inner = plt.Circle(center, r_start, fill=False, color='cyan', linewidth=2)
    circle_outer = plt.Circle(center, r_end, fill=False, color='magenta', linewidth=2)
    circle_container = plt.Circle(center, radius, fill=False, color='green', linewidth=2)
    ax1.add_patch(circle_inner)
    ax1.add_patch(circle_outer)
    ax1.add_patch(circle_container)
    ax1.plot(center[0], center[1], 'r+', markersize=15, markeredgewidth=2)
    ax1.set_title(r'$\bf{(a)}$ Detected Circle & Analysis Region', fontsize=12)
    ax1.axis('off')
    
    # 2. Intensity grid (polar heatmap)
    ax2 = fig.add_subplot(2, 3, 2, projection='polar')
    n_theta, n_r = intensity_grid.shape
    theta = np.linspace(0, 2*np.pi, n_theta)
    r = np.linspace(r_start, r_end, n_r)
    R, Theta = np.meshgrid(r, theta)
    ax2.pcolormesh(Theta, R, intensity_grid, cmap='gray', shading='auto')
    ax2.set_title(r'$\bf{(b)}$ Intensity Grid (Polar)', fontsize=12)
    ax2.set_ylim(0, r_end * 1.1)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    
    # 3. Sample intensity profile (middle radial bin)
    ax3 = fig.add_subplot(2, 3, 3)
    mid_bin = n_r // 2
    angles = np.linspace(0, 360, n_theta)
    ax3.plot(angles, intensity_grid[:, mid_bin], 'b-', linewidth=0.8)
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Intensity')
    ax3.set_title('$\\bf{(c)}$'  ' Intensity Profile (for the middle radial bin)', fontsize=12)
    ax3.set_xlim(0, 360)
    ax3.grid(True, alpha=0.3)
    
    # 4. Individual PSDs for all radial bins
    ax4 = fig.add_subplot(2, 3, 4)
    L = np.arange(1, len(freq) // 2)
    for i in range(n_r):
        intensities = intensity_grid[:, i]
        f, psd = compute_psd(intensities)
        ax4.semilogy(f[L], psd[L], alpha=0.3, linewidth=0.5)
    ax4.set_xlabel('Frequency (finger count)')
    ax4.set_ylabel('Power (log scale)')
    ax4.set_title('$\\bf{(d)}$ Individual PSDs (all radial bins)', fontsize=12)
    ax4.set_xlim(0, 20)
    ax4.set_ylim(1e-2, 1e3)
    ax4.grid(True, alpha=0.3)
    
    # 5. Averaged PSD with peak marked
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(freq[L], avg_psd[L], 'b-', linewidth=1.5, label='Averaged PSD')
    if peak_freq:
        ax5.axvline(peak_freq, color='r', linestyle='--', linewidth=2, label=f'Peak: {peak_freq:.0f}')
    ax5.set_xlabel('Frequency (finger count)')
    ax5.set_ylabel('Power')
    ax5.set_title('$\\bf{(e)}$'  f' Averaged PSD - Detected: {peak_freq:.0f} fingers', fontsize=12)
    ax5.set_xlim(0, 20)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Top peaks table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Get top peaks
    top_peaks = get_all_peaks(freq, avg_psd, n_peaks=5)
    
    # Create table
    table_data = [['Rank', 'Finger Count', 'Power']]
    for i, (f, p) in enumerate(top_peaks):
        table_data.append([f'{i+1}', f'{f:.1f}', f'{p:.1f}'])
    
    table = ax6.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.2, 0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    # Highlight header
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Highlight top peak
    if len(top_peaks) > 0:
        for j in range(3):
            table[(1, j)].set_facecolor('#90EE90')
    
    ax6.set_title(r'$\bf{(f)}$ Top 5 Peaks', fontsize=12, y= 0.85)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def analyze_single_image(image_path: str, config: dict):
    """
    Perform complete analysis on a single image with visualization.
    Figures are automatically saved to results/figures/.
    """
    print(f"\nAnalyzing: {image_path}")
    print("=" * 60)
    
    # Load image
    image, image_gray = load_image(image_path)
    print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels")
    
    # Detect circle (automatic with manual fallback)
    center, radius, source = detect_circle_from_path(
        image_path,
        scale_percent=config.get("scale_percent", 30)
    )
    
    if center is None:
        print("[FAIL] Circle detection failed!")
        return None
    
    print(f"Circle detected ({source}): center={center}, radius={radius}px")
    
    # Get annulus info
    annulus_info = get_annulus_info(radius, config)
    print(f"Annulus: {annulus_info['inner_radius_px']}px to {annulus_info['outer_radius_px']}px")
    print(f"Resolution: {annulus_info['angular_bins']} angular x {annulus_info['radial_bins']} radial bins")
    
    # Preprocess
    blurred_gray = preprocess_image(
        image_gray,
        blur_kernel=config.get("blur_kernel", (3, 3)),
        blur_sigma=config.get("blur_sigma", 8.0)
    )
    
    # Extract intensity grid
    intensity_grid, _, _ = extract_intensity_grid(
        blurred_gray,
        center,
        radius,
        radius_start=config.get("radius_start"),
        radius_end=config.get("radius_end"),
        n_radial_bins=config.get("n_radial_bins"),
        n_angular_bins=config.get("n_angular_bins")
    )
    
    print(f"Intensity grid shape: {intensity_grid.shape}")
    
    # Compute averaged PSD and find peak
    peak_freq, avg_psd, freq = analyze_averaged_psd(intensity_grid, config)
    
    print(f"\n>>> RESULT: {peak_freq:.0f} fingers detected <<<")
    
    # Auto-save figure
    figures_folder = Path(__file__).parent.parent / "results" / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)
    original_name = Path(image_path).stem
    save_path = figures_folder / f"{original_name}_analysis.png"
    
    plot_analysis(
        image, center, radius,
        intensity_grid, avg_psd, freq, peak_freq,
        config, save_path
    )
    
    return {
        'image_path': image_path,
        'center': center,
        'radius': radius,
        'peak_freq': peak_freq,
        'intensity_grid': intensity_grid,
        'avg_psd': avg_psd,
        'freq': freq
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze a single nitrogen finger instability image")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    args = parser.parse_args()
    
    config = CONFIG.copy()
    
    # Get image path: from argument or default
    if args.image:
        image_path = args.image
    else:
        image_path = config.get("default_image", config.get("image_path_5s"))
        
        if image_path is None:
            print("Error: No image specified!")
            print("Usage: python run_single_image.py --image path/to/image.jpg")
            print("Or set 'default_image' in scripts/config.py")
            return None
        
        print(f"Using default image: {image_path}")
    
    result = analyze_single_image(image_path, config)
    
    return result


if __name__ == "__main__":
    main()
