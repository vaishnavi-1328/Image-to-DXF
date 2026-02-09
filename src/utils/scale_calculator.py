"""
Scale calculator for converting pixel coordinates to millimeters.
"""
from typing import Optional, Tuple
import numpy as np


class ScaleCalculator:
    """Calculate and apply scaling from pixels to millimeters."""

    def __init__(
        self,
        dpi: Optional[float] = None,
        pixels_per_mm: Optional[float] = None,
        reference_pixels: Optional[float] = None,
        reference_mm: Optional[float] = None
    ):
        """
        Initialize scale calculator.

        Provide one of the following:
        - dpi: Dots per inch (will convert to pixels_per_mm)
        - pixels_per_mm: Direct pixel to mm ratio
        - reference_pixels + reference_mm: Known dimension for calibration

        Args:
            dpi: Resolution in dots per inch
            pixels_per_mm: Pixels per millimeter
            reference_pixels: Known distance in pixels
            reference_mm: Corresponding distance in millimeters
        """
        self.pixels_per_mm: Optional[float] = None

        if pixels_per_mm is not None:
            self.pixels_per_mm = pixels_per_mm
        elif dpi is not None:
            # Convert DPI to pixels per mm: DPI / 25.4 (mm per inch)
            self.pixels_per_mm = dpi / 25.4
        elif reference_pixels is not None and reference_mm is not None:
            # Calculate from reference
            self.pixels_per_mm = reference_pixels / reference_mm
        else:
            # Default to 96 DPI if nothing provided
            self.pixels_per_mm = 96 / 25.4

    @property
    def dpi(self) -> float:
        """Get DPI value."""
        return self.pixels_per_mm * 25.4 if self.pixels_per_mm else 96.0

    @property
    def mm_per_pixel(self) -> float:
        """Get millimeters per pixel."""
        return 1.0 / self.pixels_per_mm if self.pixels_per_mm else 25.4 / 96

    def pixels_to_mm(self, pixels: float) -> float:
        """
        Convert pixels to millimeters.

        Args:
            pixels: Value in pixels

        Returns:
            Value in millimeters
        """
        return pixels * self.mm_per_pixel

    def mm_to_pixels(self, mm: float) -> float:
        """
        Convert millimeters to pixels.

        Args:
            mm: Value in millimeters

        Returns:
            Value in pixels
        """
        return mm * self.pixels_per_mm

    def scale_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Scale a point from pixels to millimeters.

        Args:
            point: (x, y) in pixels

        Returns:
            (x, y) in millimeters
        """
        return (
            self.pixels_to_mm(point[0]),
            self.pixels_to_mm(point[1])
        )

    def scale_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Scale a contour from pixels to millimeters.

        Args:
            contour: Nx2 array of points in pixels

        Returns:
            Nx2 array of points in millimeters
        """
        return contour * self.mm_per_pixel

    def scale_contours(self, contours: list) -> list:
        """
        Scale multiple contours from pixels to millimeters.

        Args:
            contours: List of Nx2 arrays in pixels

        Returns:
            List of Nx2 arrays in millimeters
        """
        return [self.scale_contour(c) for c in contours]

    def __repr__(self) -> str:
        return f"ScaleCalculator(dpi={self.dpi:.2f}, pixels_per_mm={self.pixels_per_mm:.4f})"


def calculate_scale_from_dpi(dpi: float) -> float:
    """
    Calculate pixels per mm from DPI.

    Args:
        dpi: Dots per inch

    Returns:
        Pixels per millimeter
    """
    return dpi / 25.4


def calculate_dpi_from_scale(pixels_per_mm: float) -> float:
    """
    Calculate DPI from pixels per mm.

    Args:
        pixels_per_mm: Pixels per millimeter

    Returns:
        DPI
    """
    return pixels_per_mm * 25.4
