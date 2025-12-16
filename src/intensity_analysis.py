"""
Intensity Analysis Module

Functions for extracting angular intensity profiles from circular regions.
Handles coordinate transformation and 2D binning for spectral analysis.
"""

import numpy as np
from scipy import stats


def compute_polar_coordinates(image_shape: tuple, center: tuple) -> tuple:
    """
    Compute polar coordinates (r, theta) for all pixels in an image.
    
    Parameters
    ----------
    image_shape : tuple
        Shape of the image (height, width)
    center : tuple
        Circle center (x, y) in pixel coordinates
    
    Returns
    -------
    tuple
        (r, theta_deg) arrays with same shape as image
    """
    y, x = np.indices(image_shape)
    
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    theta = np.arctan2(y - center[1], x - center[0])
    theta_deg = (np.degrees(theta) + 360) % 360
    
    return r, theta_deg


def extract_intensity_grid(
    image_gray: np.ndarray,
    center: tuple,
    radius: int,
    radius_start: float = 0.6,
    radius_end: float = 0.9,
    n_radial_bins: int = 16,
    n_angular_bins: int = 360
) -> tuple:
    """
    Extract 2D intensity grid by binning annular region by (theta, r).
    
    Parameters
    ----------
    image_gray : np.ndarray
        Grayscale image
    center : tuple
        Circle center (x, y)
    radius : int
        Container radius in pixels
    radius_start : float
        Inner radius as fraction of container radius
    radius_end : float
        Outer radius as fraction of container radius
    n_radial_bins : int
        Number of radial bins
    n_angular_bins : int
        Number of angular bins
    
    Returns
    -------
    tuple
        (intensity_grid, theta_centers, r_centers)
    """
    r_start = int(radius * radius_start)
    r_end = int(radius * radius_end)
    
    # Compute polar coordinates
    r, theta_deg = compute_polar_coordinates(image_gray.shape, center)
    
    # Mask for annular region
    annulus_mask = (r >= r_start) & (r <= r_end)
    
    # Extract values within annulus
    r_annulus = r[annulus_mask]
    theta_annulus = theta_deg[annulus_mask]
    intensity_annulus = image_gray[annulus_mask]
    
    # Define bin edges
    theta_bins = np.linspace(0, 360, n_angular_bins + 1)
    r_bins = np.linspace(r_start, r_end, n_radial_bins + 1)
    
    # 2D binning
    intensity_grid, theta_edges, r_edges, _ = stats.binned_statistic_2d(
        theta_annulus,
        r_annulus,
        intensity_annulus,
        statistic='mean',
        bins=[theta_bins, r_bins]
    )
    
    # Handle NaN values
    intensity_grid = np.nan_to_num(intensity_grid, nan=0)
    
    # Compute bin centers
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    
    return intensity_grid, theta_centers, r_centers


def get_annulus_info(radius: int, config: dict) -> dict:
    """
    Get information about the annular sampling region.
    
    Parameters
    ----------
    radius : int
        Container radius in pixels
    config : dict
        Configuration dictionary
    
    Returns
    -------
    dict
        Dictionary with annulus information
    """
    r_start = int(radius * config.get("radius_start", 0.6))
    r_end = int(radius * config.get("radius_end", 0.9))
    n_radial = config.get("n_radial_bins", 16)
    n_angular = config.get("n_angular_bins", 360)
    
    return {
        "inner_radius_px": r_start,
        "outer_radius_px": r_end,
        "annulus_width_px": r_end - r_start,
        "radial_bins": n_radial,
        "angular_bins": n_angular,
        "radial_resolution_px": (r_end - r_start) / n_radial,
        "angular_resolution_deg": 360 / n_angular,
    }
