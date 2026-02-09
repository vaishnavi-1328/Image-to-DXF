"""
Contour extraction from edge maps using OpenCV.
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ContourInfo:
    """Information about an extracted contour."""
    points: np.ndarray  # Nx2 array of (x, y) points
    area: float  # Area in pixels
    is_hole: bool  # True if this is a hole/interior contour
    parent_idx: Optional[int]  # Index of parent contour (None if top-level)
    children_idx: List[int]  # Indices of child contours


class ContourExtractor:
    """
    Extract contours from edge maps using OpenCV's findContours.
    """

    def __init__(self, config: dict):
        """
        Initialize contour extractor with configuration.

        Args:
            config: Configuration dictionary with contour_extraction parameters
        """
        self.config = config
        self.min_area = config.get('min_contour_area_pixels', 100)
        self.retrieval_mode_str = config.get('retrieval_mode', 'RETR_CCOMP')
        self.approximation_mode_str = config.get('approximation_mode', 'CHAIN_APPROX_SIMPLE')

        # Map string to OpenCV constant
        self.retrieval_mode = getattr(cv2, self.retrieval_mode_str)
        self.approximation_mode = getattr(cv2, self.approximation_mode_str)

    def extract_contours(
        self,
        edge_map: np.ndarray
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """
        Extract contours from edge map.

        Args:
            edge_map: Binary edge map (0 or 255)

        Returns:
            Tuple of (list of contours, hierarchy array)
            Each contour is an Nx2 array of (x, y) points
        """
        # Find contours
        contours, hierarchy = cv2.findContours(
            edge_map,
            self.retrieval_mode,
            self.approximation_mode
        )

        # Convert contours to Nx2 format (remove extra dimension)
        contours = [np.squeeze(cnt, axis=1) for cnt in contours if len(cnt) >= 3]

        # Filter by area if contours have valid shape
        filtered_contours = []
        for cnt in contours:
            if len(cnt.shape) == 2 and cnt.shape[0] >= 3:
                area = cv2.contourArea(cnt)
                if area >= self.min_area:
                    filtered_contours.append(cnt)

        return filtered_contours, hierarchy

    def extract_contours_with_info(
        self,
        edge_map: np.ndarray
    ) -> List[ContourInfo]:
        """
        Extract contours with full hierarchy information.

        Args:
            edge_map: Binary edge map

        Returns:
            List of ContourInfo objects with hierarchy
        """
        contours, hierarchy = cv2.findContours(
            edge_map,
            self.retrieval_mode,
            self.approximation_mode
        )

        if hierarchy is None:
            return []

        # Hierarchy format: [Next, Previous, First_Child, Parent]
        hierarchy = hierarchy[0]  # Remove outer dimension

        contour_infos = []

        for idx, cnt in enumerate(contours):
            # Skip if too few points
            if len(cnt) < 3:
                continue

            # Convert to Nx2 format
            points = np.squeeze(cnt, axis=1)

            # Skip if wrong shape
            if len(points.shape) != 2 or points.shape[0] < 3:
                continue

            # Calculate area
            area = cv2.contourArea(cnt)

            # Skip if too small
            if area < self.min_area:
                continue

            # Get hierarchy info
            h = hierarchy[idx]
            parent_idx = h[3] if h[3] >= 0 else None

            # Determine if hole (has parent)
            is_hole = parent_idx is not None

            # Find children
            children_idx = []
            first_child = h[2]
            if first_child >= 0:
                # Traverse children
                child_idx = first_child
                while child_idx >= 0:
                    children_idx.append(child_idx)
                    child_idx = hierarchy[child_idx][0]  # Next sibling

            contour_info = ContourInfo(
                points=points,
                area=area,
                is_hole=is_hole,
                parent_idx=parent_idx,
                children_idx=children_idx
            )

            contour_infos.append(contour_info)

        return contour_infos

    def separate_outer_and_holes(
        self,
        edge_map: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Separate outer contours from holes.

        Args:
            edge_map: Binary edge map

        Returns:
            Tuple of (outer_contours, hole_contours)
        """
        contour_infos = self.extract_contours_with_info(edge_map)

        outer_contours = []
        hole_contours = []

        for info in contour_infos:
            if info.is_hole:
                hole_contours.append(info.points)
            else:
                outer_contours.append(info.points)

        return outer_contours, hole_contours

    def filter_by_area(
        self,
        contours: List[np.ndarray],
        min_area: Optional[float] = None,
        max_area: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Filter contours by area.

        Args:
            contours: List of contours
            min_area: Minimum area (pixels), None = use config
            max_area: Maximum area (pixels), None = no limit

        Returns:
            Filtered contours
        """
        if min_area is None:
            min_area = self.min_area

        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area:
                if max_area is None or area <= max_area:
                    filtered.append(cnt)

        return filtered

    def get_contour_properties(
        self,
        contour: np.ndarray
    ) -> dict:
        """
        Get properties of a contour.

        Args:
            contour: Nx2 contour array

        Returns:
            Dictionary with properties
        """
        # Ensure correct shape for OpenCV functions
        cnt_reshaped = contour.reshape(-1, 1, 2).astype(np.int32)

        area = cv2.contourArea(cnt_reshaped)
        perimeter = cv2.arcLength(cnt_reshaped, closed=True)

        # Bounding box
        x, y, w, h = cv2.boundingRect(cnt_reshaped)

        # Centroid
        M = cv2.moments(cnt_reshaped)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx, cy = x + w/2, y + h/2

        # Compactness (circularity)
        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter * perimeter)
        else:
            compactness = 0

        return {
            'area': area,
            'perimeter': perimeter,
            'bounding_box': (x, y, w, h),
            'centroid': (cx, cy),
            'compactness': compactness,
            'num_points': len(contour)
        }

    def visualize_contours(
        self,
        image: np.ndarray,
        contours: List[np.ndarray],
        outer_color: Tuple[int, int, int] = (0, 255, 0),
        hole_color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw contours on image.

        Args:
            image: Base image (BGR)
            contours: List of contours
            outer_color: Color for outer contours (BGR)
            hole_color: Color for holes (BGR)
            thickness: Line thickness

        Returns:
            Image with drawn contours
        """
        # Convert to BGR if needed
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        # Draw contours (all in same color for now)
        for cnt in contours:
            # Reshape for OpenCV
            cnt_reshaped = cnt.reshape(-1, 1, 2).astype(np.int32)
            cv2.drawContours(vis, [cnt_reshaped], 0, outer_color, thickness)

        return vis

    def __repr__(self) -> str:
        return (
            f"ContourExtractor(min_area={self.min_area}, "
            f"mode={self.retrieval_mode_str})"
        )
