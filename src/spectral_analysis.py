"""
Spectral Analysis Module

Functions for computing Power Spectral Density (PSD) and detecting
dominant frequencies in angular intensity profiles.
"""

import numpy as np
from scipy.signal import detrend, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def highpass_filter(
    signal: np.ndarray,
    cutoff_freq: float = 3,
    sampling_rate: int = 360,
    order: int = 3
) -> np.ndarray:
    """
    Apply high-pass Butterworth filter to remove low-frequency components.
    
    This filter removes DC leakage and slow variations that can mask
    the true finger signal in the PSD.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (angular intensity profile)
    cutoff_freq : float
        Cutoff frequency in cycles per 360° (default: 3)
        Frequencies below this are attenuated.
    sampling_rate : int
        Number of samples (default: 360 for angular bins)
    order : int
        Filter order (default: 3)
    
    Returns
    -------
    np.ndarray
        Filtered signal
    
    Notes
    -----
    The signal is padded with wrapped copies to handle the circular
    boundary condition (0° connects to 360°).
    """
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # Ensure valid cutoff range
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.95
    if normalized_cutoff <= 0:
        normalized_cutoff = 0.01
    
    # Design Butterworth filter
    b, a = butter(order, normalized_cutoff, btype='high')
    
    # Pad signal for circular boundary handling
    n = len(signal)
    padded = np.concatenate([signal, signal, signal])
    
    # Apply zero-phase filtering
    filtered_padded = filtfilt(b, a, padded)
    
    # Extract middle portion
    filtered = filtered_padded[n:2*n]
    
    return filtered


def compute_psd(
    intensities: np.ndarray,
    apply_highpass: bool = True,
    highpass_cutoff: float = 3
) -> tuple:
    """
    Compute Power Spectral Density from intensity profile.
    
    Parameters
    ----------
    intensities : np.ndarray
        Angular intensity profile
    apply_highpass : bool
        Whether to apply high-pass filter (default: True)
    highpass_cutoff : float
        High-pass cutoff frequency (default: 3)
    
    Returns
    -------
    tuple
        (frequencies, psd) arrays
        - frequencies: in cycles per 360° (= finger count)
        - psd: power spectral density
    
    Examples
    --------
    >>> freq, psd = compute_psd(intensities)
    >>> dominant_freq = freq[np.argmax(psd[1:len(freq)//2]) + 1]
    """
    n = len(intensities)
    
    # Preprocessing: remove mean and linear trend
    y = detrend(intensities - np.mean(intensities))
    
    # Apply high-pass filter to remove DC leakage
    if apply_highpass:
        y = highpass_filter(y, cutoff_freq=highpass_cutoff, sampling_rate=n)
    
    # Compute FFT
    fhat = np.fft.fft(y, n)
    
    # Compute power spectral density
    psd = (1/n) * (fhat * np.conj(fhat)).real
    
    # Frequency axis (cycles per 360°)
    dx = 360 / n  # degrees per sample
    freq_per_degree = (1 / (dx * n)) * np.arange(n)
    freq_per_360 = freq_per_degree * 360  # Convert to cycles per full circle
    
    return freq_per_360, psd


def find_dominant_peak(
    freq: np.ndarray,
    psd: np.ndarray,
    freq_min: float = 4,
    freq_max: float = 15
) -> tuple:
    """
    Find the dominant frequency (highest power) within a specified range.
    
    Parameters
    ----------
    freq : np.ndarray
        Frequency array from compute_psd()
    psd : np.ndarray
        PSD array from compute_psd()
    freq_min : float
        Minimum frequency to search (default: 4)
    freq_max : float
        Maximum frequency to search (default: 15)
    
    Returns
    -------
    tuple
        (peak_frequency, peak_power) or (None, None) if no peak found
    
    Notes
    -----
    The frequency directly corresponds to finger count:
    - peak_freq = 6 means 6 fingers around the circle
    """
    n = len(freq)
    
    # Only search positive frequencies (first half of spectrum)
    L = np.arange(1, n // 2)
    
    # Apply frequency range mask
    mask = (freq[L] >= freq_min) & (freq[L] <= freq_max)
    search_indices = L[mask]
    
    if len(search_indices) == 0:
        return None, None
    
    # Find maximum power
    psd_search = psd[search_indices]
    max_idx = np.argmax(psd_search)
    
    peak_freq = freq[search_indices[max_idx]]
    peak_power = psd_search[max_idx]
    
    return peak_freq, peak_power


def analyze_averaged_psd(
    intensity_grid: np.ndarray,
    config: dict
) -> tuple:
    """
    Compute PSD for each radial bin, average them, and find dominant peak.
    
    This is the main analysis function that provides a robust estimate
    of finger count by averaging PSDs across multiple radii.
    
    Parameters
    ----------
    intensity_grid : np.ndarray
        Intensity grid from extract_intensity_grid(), shape (n_angular, n_radial)
    config : dict
        Configuration dictionary with keys:
        - smooth_sigma: Gaussian smoothing sigma (default: 2)
        - freq_min: minimum search frequency (default: 4)
        - freq_max: maximum search frequency (default: 15)
    
    Returns
    -------
    tuple
        (peak_frequency, averaged_psd, freq_array)
    
    Examples
    --------
    >>> grid = extract_intensity_grid(image, center, radius, config)
    >>> finger_count, avg_psd, freq = analyze_averaged_psd(grid, config)
    >>> print(f"Detected {finger_count:.0f} fingers")
    """
    n_angular, n_radial = intensity_grid.shape
    smooth_sigma = config.get("smooth_sigma", 2)
    
    all_psds = []
    freq = None
    
    # Compute PSD for each radial bin
    for i in range(n_radial):
        intensities = intensity_grid[:, i]
        
        # Apply smoothing
        if smooth_sigma > 0:
            intensities = gaussian_filter1d(intensities, sigma=smooth_sigma, mode='wrap')
        
        # Compute PSD
        freq, psd = compute_psd(intensities)
        all_psds.append(psd)
    
    # Average all PSDs
    avg_psd = np.mean(all_psds, axis=0)
    
    # Find dominant peak
    peak_freq, peak_power = find_dominant_peak(
        freq, avg_psd,
        freq_min=config.get("freq_min", 4),
        freq_max=config.get("freq_max", 15)
    )
    
    return peak_freq, avg_psd, freq


def get_all_peaks(freq, psd, n_peaks=5, freq_min=1, freq_max=50):
    """Find local maxima (points higher than both neighbors) in PSD."""
    
    peak_indices, _ = find_peaks(psd)
    
    # Filter by frequency range
    valid = [(freq[i], psd[i]) for i in peak_indices if freq_min <= freq[i] <= freq_max]
    
    # Sort by power and return top n
    valid.sort(key=lambda x: x[1], reverse=True)
    return valid[:n_peaks]