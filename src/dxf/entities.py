"""
Helper functions for creating DXF entities from contour data.
"""
import numpy as np
from typing import List, Tuple, Union
import ezdxf


def points_to_dxf_format(points: np.ndarray) -> List[Tuple[float, float]]:
    """
    Convert Nx2 numpy array to list of (x, y) tuples for DXF.

    Args:
        points: Nx2 numpy array of points

    Returns:
        List of (x, y) tuples with float coordinates
    """
    if isinstance(points, list):
        # Already a list, ensure tuples of floats
        return [(float(x), float(y)) for x, y in points]

    # Handle numpy array
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError(f"Points must be Nx2 array, got shape {points.shape}")

    # Convert to list of tuples
    return [(float(x), float(y)) for x, y in points]


def create_lwpolyline(
    msp,
    points: Union[np.ndarray, List[Tuple[float, float]]],
    closed: bool = True,
    layer: str = "CutLines",
    dxfattribs: dict = None
) -> ezdxf.entities.LWPolyline:
    """
    Create a lightweight polyline entity.

    Args:
        msp: Modelspace from ezdxf document
        points: Nx2 numpy array or list of (x, y) tuples (in millimeters)
        closed: Whether the polyline should be closed
        layer: Layer name
        dxfattribs: Additional DXF attributes

    Returns:
        LWPolyline entity
    """
    # Convert points to DXF format
    dxf_points = points_to_dxf_format(points)

    # Ensure closed path has matching first/last point
    if closed and len(dxf_points) > 0:
        first = dxf_points[0]
        last = dxf_points[-1]

        # Check if already closed (within small tolerance)
        if not (abs(first[0] - last[0]) < 1e-6 and abs(first[1] - last[1]) < 1e-6):
            # Add first point at end to close
            dxf_points.append(first)

    # Prepare attributes
    if dxfattribs is None:
        dxfattribs = {}

    dxfattribs['layer'] = layer

    # Create polyline (DXF polylines are always 2D, Z is implicitly 0)
    polyline = msp.add_lwpolyline(dxf_points, dxfattribs=dxfattribs)

    # Set closed flag
    polyline.closed = closed

    return polyline


def create_circle(
    msp,
    center: Tuple[float, float],
    radius: float,
    layer: str = "CutLines",
    dxfattribs: dict = None
) -> ezdxf.entities.Circle:
    """
    Create a circle entity.

    Args:
        msp: Modelspace from ezdxf document
        center: (x, y) center coordinates in millimeters
        radius: Radius in millimeters
        layer: Layer name
        dxfattribs: Additional DXF attributes

    Returns:
        Circle entity
    """
    # Prepare attributes
    if dxfattribs is None:
        dxfattribs = {}

    dxfattribs['layer'] = layer

    # Create circle (at Z=0)
    circle = msp.add_circle(
        center=(float(center[0]), float(center[1]), 0.0),
        radius=float(radius),
        dxfattribs=dxfattribs
    )

    return circle


def create_arc(
    msp,
    center: Tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    layer: str = "CutLines",
    dxfattribs: dict = None
) -> ezdxf.entities.Arc:
    """
    Create an arc entity.

    Args:
        msp: Modelspace from ezdxf document
        center: (x, y) center coordinates in millimeters
        radius: Radius in millimeters
        start_angle: Start angle in degrees
        end_angle: End angle in degrees
        layer: Layer name
        dxfattribs: Additional DXF attributes

    Returns:
        Arc entity
    """
    # Prepare attributes
    if dxfattribs is None:
        dxfattribs = {}

    dxfattribs['layer'] = layer

    # Create arc (at Z=0)
    arc = msp.add_arc(
        center=(float(center[0]), float(center[1]), 0.0),
        radius=float(radius),
        start_angle=float(start_angle),
        end_angle=float(end_angle),
        dxfattribs=dxfattribs
    )

    return arc


def ensure_closed_path(points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """
    Ensure a path is closed by adding first point at end if needed.

    Args:
        points: Nx2 numpy array of points
        tolerance: Distance tolerance for considering points equal

    Returns:
        Closed path (Nx2 numpy array)
    """
    if len(points) < 2:
        return points

    # Check distance between first and last point
    dist = np.linalg.norm(points[0] - points[-1])

    if dist < tolerance:
        # Already closed
        return points

    # Close by adding first point at end
    return np.vstack([points, points[0:1]])


def validate_points(points: Union[np.ndarray, List]) -> None:
    """
    Validate points array for DXF generation.

    Args:
        points: Points to validate

    Raises:
        ValueError: If points are invalid
    """
    if isinstance(points, np.ndarray):
        if len(points.shape) != 2 or points.shape[1] != 2:
            raise ValueError(f"Points must be Nx2 array, got shape {points.shape}")

        if len(points) < 3:
            raise ValueError(f"Need at least 3 points for a closed path, got {len(points)}")

        # Check for NaN or Inf
        if np.any(~np.isfinite(points)):
            raise ValueError("Points contain NaN or Inf values")

    elif isinstance(points, list):
        if len(points) < 3:
            raise ValueError(f"Need at least 3 points for a closed path, got {len(points)}")

        for pt in points:
            if len(pt) != 2:
                raise ValueError(f"Each point must have 2 coordinates, got {len(pt)}")

            if not all(np.isfinite([pt[0], pt[1]])):
                raise ValueError("Points contain NaN or Inf values")

    else:
        raise ValueError(f"Points must be numpy array or list, got {type(points)}")
