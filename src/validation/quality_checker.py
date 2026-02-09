"""
Quality validation for contours before DXF generation.
"""
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """Validation message severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationMessage:
    """A validation message."""
    severity: Severity
    message: str
    contour_idx: int = None


class QualityChecker:
    """
    Validate contours before DXF generation.
    """

    def __init__(self, config: dict):
        """
        Initialize quality checker with configuration.

        Args:
            config: Validation configuration dictionary
        """
        self.config = config
        self.min_feature_size_mm = config.get('min_feature_size_mm', 1.0)
        self.max_entity_count = config.get('max_entity_count', 10000)
        self.allow_open_paths = config.get('allow_open_paths', False)
        self.check_self_intersections = config.get('check_self_intersections', True)

    def validate_contours(
        self,
        contours: List[np.ndarray]
    ) -> Tuple[bool, List[ValidationMessage], List[ValidationMessage]]:
        """
        Validate all contours.

        Args:
            contours: List of Nx2 contour arrays (in millimeters)

        Returns:
            Tuple of (is_valid, warnings, errors)
            - is_valid: True if no errors (warnings are OK)
            - warnings: List of warning messages
            - errors: List of error messages
        """
        warnings = []
        errors = []

        # Check total entity count
        if len(contours) > self.max_entity_count:
            errors.append(ValidationMessage(
                Severity.ERROR,
                f"Too many contours: {len(contours)} (max: {self.max_entity_count})",
                None
            ))

        # Validate each contour
        for idx, contour in enumerate(contours):
            messages = self.validate_single_contour(contour, idx)

            for msg in messages:
                if msg.severity == Severity.ERROR:
                    errors.append(msg)
                elif msg.severity == Severity.WARNING:
                    warnings.append(msg)

        # Check for duplicates
        duplicate_msgs = self.check_duplicate_paths(contours)
        for msg in duplicate_msgs:
            if msg.severity == Severity.WARNING:
                warnings.append(msg)

        is_valid = len(errors) == 0
        return is_valid, warnings, errors

    def validate_single_contour(
        self,
        contour: np.ndarray,
        idx: int
    ) -> List[ValidationMessage]:
        """
        Validate a single contour.

        Args:
            contour: Nx2 contour array (in millimeters)
            idx: Contour index

        Returns:
            List of validation messages
        """
        messages = []

        # Check point count
        point_count_msg = self.check_point_count(contour, idx)
        if point_count_msg:
            messages.append(point_count_msg)

        # Check for invalid coordinates
        coord_msg = self.check_coordinates(contour, idx)
        if coord_msg:
            messages.append(coord_msg)

        # Check if closed
        if not self.allow_open_paths:
            closed_msg = self.check_closed_path(contour, idx)
            if closed_msg:
                messages.append(closed_msg)

        # Check minimum size
        size_msg = self.check_minimum_size(contour, self.min_feature_size_mm, idx)
        if size_msg:
            messages.append(size_msg)

        # Check self-intersection (optional, can be slow)
        if self.check_self_intersections and len(contour) > 3:
            intersect_msg = self.check_self_intersection(contour, idx)
            if intersect_msg:
                messages.append(intersect_msg)

        return messages

    def check_point_count(self, contour: np.ndarray, idx: int) -> ValidationMessage:
        """
        Check if contour has valid number of points.

        Args:
            contour: Contour array
            idx: Contour index

        Returns:
            ValidationMessage or None
        """
        n_points = len(contour)

        if n_points < 3:
            return ValidationMessage(
                Severity.ERROR,
                f"Contour #{idx}: Too few points ({n_points}, minimum: 3)",
                idx
            )

        if n_points > 10000:
            return ValidationMessage(
                Severity.WARNING,
                f"Contour #{idx}: Very high point count ({n_points}), may be slow",
                idx
            )

        if n_points > 5000:
            return ValidationMessage(
                Severity.INFO,
                f"Contour #{idx}: High point count ({n_points})",
                idx
            )

        return None

    def check_coordinates(self, contour: np.ndarray, idx: int) -> ValidationMessage:
        """
        Check for invalid coordinates (NaN, Inf).

        Args:
            contour: Contour array
            idx: Contour index

        Returns:
            ValidationMessage or None
        """
        if np.any(~np.isfinite(contour)):
            return ValidationMessage(
                Severity.ERROR,
                f"Contour #{idx}: Contains NaN or Inf coordinates",
                idx
            )

        return None

    def check_closed_path(
        self,
        contour: np.ndarray,
        idx: int,
        tolerance: float = 0.01
    ) -> ValidationMessage:
        """
        Check if path is closed (first point == last point).

        Args:
            contour: Contour array
            idx: Contour index
            tolerance: Distance tolerance in mm

        Returns:
            ValidationMessage or None
        """
        if len(contour) < 2:
            return None

        # Calculate distance between first and last point
        dist = np.linalg.norm(contour[0] - contour[-1])

        if dist > tolerance:
            if dist > 1.0:  # > 1mm is an error
                return ValidationMessage(
                    Severity.ERROR,
                    f"Contour #{idx}: Path not closed (gap: {dist:.2f} mm)",
                    idx
                )
            else:  # Small gap is a warning
                return ValidationMessage(
                    Severity.WARNING,
                    f"Contour #{idx}: Path not perfectly closed (gap: {dist:.3f} mm)",
                    idx
                )

        return None

    def check_minimum_size(
        self,
        contour: np.ndarray,
        min_size_mm: float,
        idx: int
    ) -> ValidationMessage:
        """
        Check if feature meets minimum size requirements.

        Args:
            contour: Contour array
            idx: Contour index
            min_size_mm: Minimum size in millimeters

        Returns:
            ValidationMessage or None
        """
        # Calculate bounding box
        min_x, min_y = contour.min(axis=0)
        max_x, max_y = contour.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y

        min_dim = min(width, height)

        if min_dim < 0.5:  # < 0.5mm is likely too small
            return ValidationMessage(
                Severity.WARNING,
                f"Contour #{idx}: Very small feature ({width:.2f} x {height:.2f} mm)",
                idx
            )

        if min_dim < min_size_mm:
            return ValidationMessage(
                Severity.INFO,
                f"Contour #{idx}: Feature smaller than {min_size_mm}mm ({width:.2f} x {height:.2f} mm)",
                idx
            )

        return None

    def check_self_intersection(
        self,
        contour: np.ndarray,
        idx: int
    ) -> ValidationMessage:
        """
        Check for self-intersecting paths (simple check).

        This is a simplified check that looks for crossing line segments.

        Args:
            contour: Contour array
            idx: Contour index

        Returns:
            ValidationMessage or None
        """
        # Simple bounding box check - if segments are far apart, they can't intersect
        # This is a basic check; full intersection detection would be more complex

        n = len(contour)
        if n < 4:
            return None

        # Check a few segments (not exhaustive for performance)
        # For full check, would need to test all segment pairs
        for i in range(0, min(n-1, 100)):  # Check first 100 segments
            p1, p2 = contour[i], contour[(i+1) % n]

            for j in range(i+2, min(i+100, n-1)):
                if j == (i-1) % n or j == (i+1) % n:
                    continue  # Skip adjacent segments

                p3, p4 = contour[j], contour[(j+1) % n]

                if self._segments_intersect(p1, p2, p3, p4):
                    return ValidationMessage(
                        Severity.ERROR,
                        f"Contour #{idx}: Self-intersecting path detected",
                        idx
                    )

        return None

    def _segments_intersect(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p4: np.ndarray
    ) -> bool:
        """
        Check if two line segments intersect.

        Uses cross product method.

        Args:
            p1, p2: First segment endpoints
            p3, p4: Second segment endpoints

        Returns:
            True if segments intersect
        """
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

    def check_duplicate_paths(
        self,
        contours: List[np.ndarray],
        tolerance: float = 0.01
    ) -> List[ValidationMessage]:
        """
        Check for duplicate contours.

        Args:
            contours: List of contours
            tolerance: Distance tolerance in mm

        Returns:
            List of validation messages
        """
        messages = []
        n = len(contours)

        # Only check if reasonable number of contours
        if n > 1000:
            return messages

        for i in range(n):
            for j in range(i+1, n):
                if len(contours[i]) != len(contours[j]):
                    continue

                # Check if all points are close
                dists = np.linalg.norm(contours[i] - contours[j], axis=1)
                if np.all(dists < tolerance):
                    messages.append(ValidationMessage(
                        Severity.INFO,
                        f"Contours #{i} and #{j} are duplicates",
                        None
                    ))

        return messages

    def generate_report(
        self,
        warnings: List[ValidationMessage],
        errors: List[ValidationMessage]
    ) -> str:
        """
        Generate human-readable validation report.

        Args:
            warnings: List of warning messages
            errors: List of error messages

        Returns:
            Formatted report string
        """
        report = []
        report.append("="*60)
        report.append("VALIDATION REPORT")
        report.append("="*60)

        if len(errors) == 0 and len(warnings) == 0:
            report.append("✓ All validations passed!")
            report.append("")
            return "\n".join(report)

        if errors:
            report.append(f"\n✗ ERRORS ({len(errors)}):")
            for msg in errors:
                report.append(f"  • {msg.message}")

        if warnings:
            report.append(f"\n⚠ WARNINGS ({len(warnings)}):")
            for msg in warnings:
                report.append(f"  • {msg.message}")

        report.append("")
        report.append("="*60)

        if errors:
            report.append("Status: FAILED - Please fix errors before generating DXF")
        else:
            report.append("Status: PASSED with warnings - DXF generation can proceed")

        report.append("="*60)
        report.append("")

        return "\n".join(report)

    def __repr__(self) -> str:
        return (
            f"QualityChecker(min_feature_size={self.min_feature_size_mm}mm, "
            f"max_entities={self.max_entity_count})"
        )
