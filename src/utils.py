"""
Utility Functions

Helper functions for image loading, preprocessing, and results formatting.
"""

import cv2
import numpy as np
from pathlib import Path


def load_image(image_path: str) -> tuple:
    """
    Load image and convert to grayscale.
    
    Parameters
    ----------
    image_path : str
        Path to image file
    
    Returns
    -------
    tuple
        (image_bgr, image_gray)
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image, image_gray


def preprocess_image(
    image_gray: np.ndarray,
    blur_kernel: tuple = (3, 3),
    blur_sigma: float = 8.0
) -> np.ndarray:
    """
    Preprocess grayscale image with Gaussian blur.
    
    Parameters
    ----------
    image_gray : np.ndarray
        Grayscale image
    blur_kernel : tuple
        Gaussian blur kernel size
    blur_sigma : float
        Gaussian blur sigma
    
    Returns
    -------
    np.ndarray
        Blurred image
    """
    blurred = cv2.GaussianBlur(image_gray, blur_kernel, blur_sigma)
    return blurred


def create_results_dataframe(results: list):
    """
    Convert results list to formatted pandas DataFrame.
    
    Parameters
    ----------
    results : list
        List of result dictionaries
    
    Returns
    -------
    pd.DataFrame
        Formatted results DataFrame
    """
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    if 'Peak_5s' in df.columns:
        df['Peak_5s'] = df['Peak_5s'].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "-"
        )
    if 'Peak_10s' in df.columns:
        df['Peak_10s'] = df['Peak_10s'].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "-"
        )
    
    df = df.rename(columns={
        'Peak_5s': 'Fingers (5s)',
        'Peak_10s': 'Fingers (10s)'
    })
    
    return df


def print_results_table(df, title: str = "RESULTS SUMMARY"):
    """
    Print formatted results table to console.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    title : str
        Table title
    """
    width = 60
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)
    print()
    print(df.to_string(index=False))
    print()
    print("=" * width)
