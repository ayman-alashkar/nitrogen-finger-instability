"""
Circle Detection Module

Functions for detecting the circular container boundary in experimental images.
Uses Hough Circle Transform with manual fallback (click 3 points).
"""

import cv2
import numpy as np

# Global variable for mouse click points
click_points = []


def rescale_image(image, scale_percent):
    """Returns a resized image based on a scale percentage."""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def handle_mouse_click(event, x, y, flags, param):
    """Callback function to record mouse clicks."""
    global click_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(click_points) < 3:
            click_points.append((x, y))
            print(f"Point {len(click_points)} added at ({x}, {y})")


def calculate_circle_from_points(points):
    """
    Calculates the center and radius of a circle defined by three points.
    Returns (None, None) if the points are collinear.
    """
    p1, p2, p3 = points
    
    D = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    if D == 0:
        return None, None

    ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / D
    uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / D
    
    center = (int(ux), int(uy))
    radius = int(np.sqrt((p1[0] - ux)**2 + (p1[1] - uy)**2))
    
    return center, radius


def detect_circle(image, scale_percent=30, window_name="Click 3 points on container rim"):
    """
    Detect circle automatically or via manual selection.
    
    1. First tries automatic Hough Circle detection
    2. If fails, opens window for manual 3-point selection
    
    Parameters
    ----------
    image : np.ndarray
        Input BGR image
    scale_percent : int
        Scale factor for processing/display (default: 30%)
    window_name : str
        Window title for manual selection
    
    Returns
    -------
    tuple
        (center, radius, source) where:
        - center is (x, y) tuple
        - radius is int
        - source is 'automatic' or 'manual' or None if failed
    """
    global click_points
    
    scale_factor = 100.0 / scale_percent
    rescaled_image = rescale_image(image, scale_percent)
    rescaled_gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
    
    # Automatic detection
    print("Attempting automatic circle detection...")
    blurred_image = cv2.medianBlur(rescaled_gray, 5)
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=30, minRadius=100, maxRadius=400)
    
    if circles is not None:
        c = circles[0, 0]
        cx_scaled, cy_scaled, r_scaled = int(c[0]), int(c[1]), int(c[2])
        center = (int(cx_scaled * scale_factor), int(cy_scaled * scale_factor))
        radius = int(r_scaled * scale_factor)
        print(f"Automatic detection successful! Center {center}, radius {radius}px.")
        return center, radius, 'automatic'
    
    # Manual fallback
    print("Automatic detection failed. Falling back to manual selection.")
    click_points = []
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, handle_mouse_click)
    print("\nPlease click on three different points on the rim of the container.")
    
    while len(click_points) < 3:
        temp_display_image = rescaled_image.copy()
        for i, p in enumerate(click_points):
            cv2.circle(temp_display_image, p, 5, (0, 255, 0), -1)
            cv2.putText(temp_display_image, str(i+1), (p[0]+10, p[1]+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window_name, temp_display_image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
            break
    cv2.destroyAllWindows()
    
    if len(click_points) < 3:
        print("Manual selection cancelled or incomplete.")
        return None, None, None
    
    original_points = [(int(p[0] * scale_factor), int(p[1] * scale_factor)) for p in click_points]
    center, radius = calculate_circle_from_points(original_points)
    
    if center:
        print(f"Manual circle calculated: center {center}, radius {radius}px.")
        return center, radius, 'manual'
    
    return None, None, None


def detect_circle_from_path(image_path, scale_percent=30):
    """
    Detect circle from image path.
    
    Parameters
    ----------
    image_path : str
        Path to the image file
    scale_percent : int
        Scale factor for processing
    
    Returns
    -------
    tuple
        (center, radius, source)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[FAIL] Could not read image: {image_path}")
        return None, None, None
    
    return detect_circle(image, scale_percent=scale_percent)
