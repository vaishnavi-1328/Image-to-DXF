"""
Path simplification using Douglas-Peucker algorithm.
"""
import numpy as np
import cv2
from typing import List, Optional


class PathSimplifier:
    """
    Simplify paths using Douglas-Peucker algorithm while preserving shape.
    """

    def __init__(self, config: dict, scale_calculator=None):
        """
        Initialize path simplifier.

        Args:
            config: Configuration dictionary with vectorization parameters
            scale_calculator: ScaleCalculator instance for pixel-to-mm conversion
        """
        self.config = config
        self.scale_calculator = scale_calculator

        # Epsilon in millimeters (will be converted to pixels)
        self.epsilon_mm = config.get('simplify_epsilon_mm', 0.05)

        # Circle detection settings
        self.detect_circles = config.get('detect_circles', True)
        self.circle_tolerance = config.get('circle_tolerance', 0.02)

    def get_epsilon_pixels(self) -> float:
        """
        Get epsilon value in pixels.

        Returns:
            Epsilon in pixels
        """
        if self.scale_calculator:
            return self.scale_calculator.mm_to_pixels(self.epsilon_mm)
        else:
            # Fallback: assume 96 DPI (3.78 pixels/mm)
            return self.epsilon_mm * 3.78

    def simplify_contour(
        self,
        contour: np.ndarray,
        epsilon: Optional[float] = None,
        closed: bool = True
    ) -> np.ndarray:
        """
        Simplify a contour using Douglas-Peucker algorithm.

        Args:
            contour: Nx2 array of points
            epsilon: Simplification epsilon (pixels), None = use config
            closed: Whether contour is closed

        Returns:
            Simplified contour (Nx2 array)
        """
        if epsilon is None:
            epsilon = self.get_epsilon_pixels()

        # Reshape for OpenCV (needs Nx1x2 format)
        contour_reshaped = contour.reshape(-1, 1, 2).astype(np.float32)

        # Apply Douglas-Peucker
        simplified = cv2.approxPolyDP(contour_reshaped, epsilon, closed)

        # Reshape back to Nx2
        simplified = np.squeeze(simplified, axis=1)

        # Ensure closed if requested
        if closed and len(simplified) > 0:
            # Check if first and last points are the same
            if not np.allclose(simplified[0], simplified[-1], atol=1e-6):
                # Close the path
                simplified = np.vstack([simplified, simplified[0:1]])

        return simplified

    def simplify_contours(
        self,
        contours: List[np.ndarray],
        epsilon: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Simplify multiple contours.

        Args:
            contours: List of contours
            epsilon: Simplification epsilon (pixels)

        Returns:
            List of simplified contours
        """
        return [self.simplify_contour(cnt, epsilon) for cnt in contours]

    def is_circular(
        self,
        contour: np.ndarray,
        tolerance: Optional[float] = None
    ) -> tuple:
        """
        Check if contour is approximately circular.

        Args:
            contour: Nx2 contour array
            tolerance: Tolerance for circularity (0-1), None = use config

        Returns:
            Tuple of (is_circular, center, radius)
        """
        if tolerance is None:
            tolerance = self.circle_tolerance

        # Need at least 5 points
        if len(contour) < 5:
            return False, None, None

        # Fit minimum enclosing circle
        contour_reshaped = contour.reshape(-1, 1, 2).astype(np.float32)
        (cx, cy), radius = cv2.minEnclosingCircle(contour_reshaped)

        # Calculate area ratio
        contour_area = cv2.contourArea(contour_reshaped)
        circle_area = np.pi * radius * radius

        if circle_area == 0:
            return False, None, None

        area_ratio = contour_area / circle_area

        # Check if close to 1.0 (circular)
        is_circular = abs(1.0 - area_ratio) < tolerance

        return is_circular, (cx, cy), radius

    def detect_and_convert_circles(
        self,
        contours: List[np.ndarray]
    ) -> List[dict]:
        """
        Detect circular contours and convert to circle primitives.

        Args:
            contours: List of contours

        Returns:
            List of dicts with 'type' ('contour' or 'circle') and data
        """
        if not self.detect_circles:
            return [{'type': 'contour', 'points': cnt} for cnt in contours]

        results = []

        for cnt in contours:
            is_circular, center, radius = self.is_circular(cnt)

            if is_circular:
                results.append({
                    'type': 'circle',
                    'center': center,
                    'radius': radius,
                    'original_points': cnt
                })
            else:
                results.append({
                    'type': 'contour',
                    'points': cnt
                })

        return results

    def ensure_closed(self, contour: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        """
        Ensure contour is closed by duplicating first point if needed.

        Args:
            contour: Nx2 contour array
            tolerance: Distance tolerance for considering points equal

        Returns:
            Closed contour
        """
        if len(contour) < 2:
            return contour

        # Check if already closed
        dist = np.linalg.norm(contour[0] - contour[-1])
        if dist < tolerance:
            return contour

        # Close by adding first point at end
        return np.vstack([contour, contour[0:1]])

    def remove_duplicate_points(
        self,
        contour: np.ndarray,
        tolerance: float = 1e-6
    ) -> np.ndarray:
        """
        Remove duplicate consecutive points.

        Args:
            contour: Nx2 contour array
            tolerance: Distance tolerance

        Returns:
            Contour without duplicates
        """
        if len(contour) < 2:
            return contour

        # Calculate distances between consecutive points
        diffs = np.linalg.norm(np.diff(contour, axis=0), axis=1)

        # Keep points where difference is above tolerance
        keep = np.concatenate([[True], diffs > tolerance])

        return contour[keep]

    def get_simplification_stats(
        self,
        original_contours: List[np.ndarray],
        simplified_contours: List[np.ndarray]
    ) -> dict:
        """
        Get statistics about simplification.

        Args:
            original_contours: Original contours
            simplified_contours: Simplified contours

        Returns:
            Dictionary with statistics
        """
        orig_total_points = sum(len(cnt) for cnt in original_contours)
        simp_total_points = sum(len(cnt) for cnt in simplified_contours)

        reduction = (orig_total_points - simp_total_points) / orig_total_points * 100 if orig_total_points > 0 else 0

        return {
            'original_contours': len(original_contours),
            'simplified_contours': len(simplified_contours),
            'original_total_points': orig_total_points,
            'simplified_total_points': simp_total_points,
            'reduction_percent': reduction,
            'avg_points_original': orig_total_points / len(original_contours) if original_contours else 0,
            'avg_points_simplified': simp_total_points / len(simplified_contours) if simplified_contours else 0
        }

    def __repr__(self) -> str:
        return f"PathSimplifier(epsilon_mm={self.epsilon_mm}, detect_circles={self.detect_circles})"
