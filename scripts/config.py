"""
Configuration Parameters

Central configuration for all analysis scripts.
Modify these parameters to adjust the analysis.
"""

CONFIG = {
    # ===================
    # Data Configuration
    # ===================
    
    # Folder containing experimental images
    "folder_path": "path/to/your/folder", # Update it with your folder path
    
    # Default image for single image analysis (used if no --image argument provided)
    "default_image":  "path/to/your/image.jpg", # Update it with your image path
    
    
    # Experimental conditions to analyze
    "diameters": ["d1", "d2", "d3", "d4"],
    "temperatures": ["T5", "T10", "T20", "T40"],
    
    # ========================
    # Analysis Parameters
    # ========================
    
    # Annular region for sampling (as fraction of container radius)
    "radius_start": 0.6,    # Inner radius: 60% of container
    "radius_end": 0.9,      # Outer radius: 90% of container
    
    # Binning resolution
    "n_radial_bins": 32,    # Number of radial bins for PSD averaging
    "n_angular_bins": 360,  # Angular resolution (1° per bin)
    
    # Smoothing
    "smooth_sigma": 2,      # Gaussian smoothing sigma (0 = no smoothing)
    
    # ========================
    # Peak Detection
    # ========================
    
    # Frequency search range (= expected finger count range)
    "freq_min": 4,          # Minimum fingers to detect
    "freq_max": 15,         # Maximum fingers to detect
    
    # High-pass filter cutoff (removes DC leakage)
    "highpass_cutoff": 4,   # Remove frequencies below this
    
    # ========================
    # Circle Detection
    # ========================
    
    # Image rescaling for faster detection
    "scale_percent": 30,    # Process at 30% resolution
    
    # Hough Circle parameters
    "hough_param1": 100,    # Canny edge threshold
    "hough_param2": 30,     # Accumulator threshold
    "min_radius": 100,      # Minimum radius (in scaled image)
    "max_radius": 400,      # Maximum radius (in scaled image)
    
    # ========================
    # Image Preprocessing
    # ========================
    
    "blur_kernel": (3, 3),
    "blur_sigma": 8.0,
    
    # ========================
    # Output
    # ========================
    
    "output_csv": "analysis_results.csv",
    "save_figures": True,
    "figures_folder": "results/figures",
}


def get_config():
    """Return a copy of the configuration dictionary."""
    return CONFIG.copy()


def print_config():
    """Print current configuration to console."""
    print("\n" + "=" * 50)
    print("CURRENT CONFIGURATION")
    print("=" * 50)
    
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    print("=" * 50 + "\n")
