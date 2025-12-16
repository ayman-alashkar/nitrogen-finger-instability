"""
Batch Analysis Script

Analyze all images in a folder and generate results table.

Usage:
    python run_batch_analysis.py
    python run_batch_analysis.py --folder /path/to/images
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.circle_detection import detect_circle_from_path
from src.intensity_analysis import extract_intensity_grid
from src.spectral_analysis import analyze_averaged_psd
from src.utils import load_image, preprocess_image, create_results_dataframe, print_results_table
from scripts.config import CONFIG, print_config


def analyze_image_with_circle(image_path: str, center: tuple, radius: int, config: dict) -> float:
    """
    Analyze a single image using provided circle parameters.
    Returns the detected finger count (peak frequency).
    """
    # Load and preprocess
    image, image_gray = load_image(image_path)
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
        radius_start=config.get("radius_start", 0.6),
        radius_end=config.get("radius_end", 0.9),
        n_radial_bins=config.get("n_radial_bins", 16),
        n_angular_bins=config.get("n_angular_bins", 360)
    )
    
    # Compute averaged PSD and find peak
    peak_freq, _, _ = analyze_averaged_psd(intensity_grid, config)
    
    return peak_freq


def batch_analyze(config: dict) -> list:
    """
    Run batch analysis on all configured images.
    
    Circle detection is done per (diameter, temperature) pair.
    The circle from _1.JPG is used for both _1 and _2 images.
    
    Returns list of result dictionaries.
    """
    folder = Path(config["folder_path"])
    all_results = []
    
    print("\n" + "=" * 80)
    print("BATCH ANALYSIS - NITROGEN FINGER INSTABILITY")
    print(f"Averaging PSDs from {int(config['radius_start']*100)}% to {int(config['radius_end']*100)}% radius")
    print(f"Using {config['n_radial_bins']} radial bins")
    print("=" * 80)
    
    for diameter in config["diameters"]:
        print(f"\n>>> Processing {diameter}...")
        
        for temp in config["temperatures"]:
            # Build file paths
            path_5s = folder / f"{diameter}_{temp}_1.JPG"
            path_10s = folder / f"{diameter}_{temp}_2.JPG"
            
            # Check existence
            if not path_5s.exists():
                print(f"    [SKIP] Missing: {path_5s.name}")
                continue
            if not path_10s.exists():
                print(f"    [SKIP] Missing: {path_10s.name}")
                continue
            
            print(f"    {diameter}_{temp}: ", end="")
            
            # Detect circle from the 5s image (automatic with manual fallback)
            # This will open a window for manual selection if automatic fails
            center, radius, source = detect_circle_from_path(
                str(path_5s),
                scale_percent=config.get("scale_percent", 30)
            )
            
            if center is None:
                print("[FAIL] Circle detection failed, skipping...")
                continue
            
            print(f"circle ({source}) ", end="")
            
            try:
                # Analyze both images using the same circle
                peak_5s = analyze_image_with_circle(str(path_5s), center, radius, config)
                peak_10s = analyze_image_with_circle(str(path_10s), center, radius, config)
                
                if peak_5s is None or peak_10s is None:
                    print("[FAIL] Analysis failed")
                    continue
                
                result = {
                    'Diameter': diameter,
                    'Temp': temp,
                    'Peak_5s': peak_5s,
                    'Peak_10s': peak_10s,
                    'Center': center,
                    'Radius': radius,
                    'Detection': source,
                }
                all_results.append(result)
                
                print(f"[OK] 5s: {peak_5s:.0f}, 10s: {peak_10s:.0f}")
                
            except Exception as e:
                print(f"[FAIL] Error: {e}")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch analyze nitrogen finger instability images")
    parser.add_argument("--folder", type=str, help="Folder containing images")
    parser.add_argument("--config", action="store_true", help="Print configuration and exit")
    args = parser.parse_args()
    
    # Use config from file
    config = CONFIG.copy()
    
    # Override folder if provided
    if args.folder:
        config["folder_path"] = args.folder
    
    # Print config if requested
    if args.config:
        print_config()
        return
    
    # Run analysis
    results = batch_analyze(config)
    
    if not results:
        print("\nNo results generated!")
        return
    
    # Create and print results table
    df = create_results_dataframe(results)
    print_results_table(df)
    
    # Save to CSV in results/ folder
    results_folder = Path(__file__).parent.parent / "results"
    results_folder.mkdir(parents=True, exist_ok=True)
    output_path = results_folder / config.get("output_csv", "analysis_results.csv")
    
    # Save raw data (before formatting)
    raw_df = pd.DataFrame(results)
    raw_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return results, df


if __name__ == "__main__":
    main()
