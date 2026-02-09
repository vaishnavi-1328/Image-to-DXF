#!/usr/bin/env python
"""
Test script for Phase 3 components (DXF generation and validation).
"""
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config_loader import load_config
from utils.logger import setup_logger
from validation.quality_checker import QualityChecker, Severity
from dxf.entities import (
    create_lwpolyline, create_circle, points_to_dxf_format,
    ensure_closed_path, validate_points
)
from dxf.generator import DXFGenerator
import ezdxf


def test_entity_helpers(logger):
    """Test DXF entity helper functions."""
    logger.info("="*60)
    logger.info("Testing DXF Entity Helpers")
    logger.info("="*60)

    # Test points_to_dxf_format
    points_np = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    dxf_points = points_to_dxf_format(points_np)

    logger.info(f"  points_to_dxf_format: {type(dxf_points)}")
    assert isinstance(dxf_points, list), "Should return list"
    assert len(dxf_points) == 4, "Should have 4 points"
    assert all(isinstance(p, tuple) for p in dxf_points), "Should be tuples"

    # Test ensure_closed_path
    open_path = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    closed_path = ensure_closed_path(open_path)

    logger.info(f"  ensure_closed_path: {len(open_path)} → {len(closed_path)} points")
    assert len(closed_path) == len(open_path) + 1, "Should add closing point"
    assert np.allclose(closed_path[0], closed_path[-1]), "Should be closed"

    # Test with already closed path
    already_closed = np.array([[0, 0], [10, 0], [10, 10], [0, 0]])
    still_closed = ensure_closed_path(already_closed)
    logger.info(f"  Already closed: {len(already_closed)} → {len(still_closed)} points")
    assert len(still_closed) == len(already_closed), "Should not add extra point"

    # Test validate_points
    try:
        validate_points(points_np)
        logger.info("  validate_points: Valid points passed ✓")
    except ValueError as e:
        logger.error(f"  validate_points failed: {e}")
        return False

    # Test invalid points
    invalid_points = np.array([[0, 0], [10, 0]])  # Too few points
    try:
        validate_points(invalid_points)
        logger.error("  validate_points should have failed for < 3 points")
        return False
    except ValueError:
        logger.info("  validate_points: Correctly rejected <3 points ✓")

    # Test create_lwpolyline
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    polyline = create_lwpolyline(msp, points_np, closed=True)
    logger.info(f"  create_lwpolyline: Created {polyline.dxftype()}")
    assert polyline.dxftype() == 'LWPOLYLINE', "Should create LWPOLYLINE"
    assert polyline.closed is True, "Should be closed"

    # Test create_circle
    circle = create_circle(msp, (50, 50), 25)
    logger.info(f"  create_circle: Created {circle.dxftype()}")
    assert circle.dxftype() == 'CIRCLE', "Should create CIRCLE"
    assert circle.dxf.radius == 25, "Should have correct radius"

    logger.info("✓ Entity helpers test passed!\n")
    return True


def test_quality_checker(logger, config):
    """Test the quality checker."""
    logger.info("="*60)
    logger.info("Testing Quality Checker")
    logger.info("="*60)

    checker = QualityChecker(config['validation'])
    logger.info(f"Checker: {checker}")

    # Create test contours
    # Good contour: properly closed rectangle
    good_rect = np.array([[0, 0], [100, 0], [100, 50], [0, 50], [0, 0]], dtype=np.float64)

    # Bad contour: open path
    open_rect = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64)

    # Small contour
    small_rect = np.array([[0, 0], [0.3, 0], [0.3, 0.2], [0, 0.2], [0, 0]], dtype=np.float64)

    # Test good contour
    logger.info("\n  Testing good contour:")
    messages = checker.validate_single_contour(good_rect, 0)
    logger.info(f"    Messages: {len(messages)}")
    for msg in messages:
        logger.info(f"      {msg.severity.value}: {msg.message}")

    # Test open path
    logger.info("\n  Testing open path:")
    messages = checker.validate_single_contour(open_rect, 1)
    logger.info(f"    Messages: {len(messages)}")
    has_error = any(msg.severity == Severity.ERROR or msg.severity == Severity.WARNING
                    for msg in messages)
    assert has_error, "Should detect open path"
    logger.info("    ✓ Open path detected")

    # Test small contour
    logger.info("\n  Testing small contour:")
    messages = checker.validate_single_contour(small_rect, 2)
    logger.info(f"    Messages: {len(messages)}")
    has_warning = any(msg.severity == Severity.WARNING for msg in messages)
    assert has_warning, "Should warn about small size"
    logger.info("    ✓ Small size detected")

    # Test full validation
    logger.info("\n  Testing full validation:")
    contours = [good_rect, small_rect]
    is_valid, warnings, errors = checker.validate_contours(contours)

    logger.info(f"    Valid: {is_valid}")
    logger.info(f"    Warnings: {len(warnings)}")
    logger.info(f"    Errors: {len(errors)}")

    # Generate report
    report = checker.generate_report(warnings, errors)
    logger.info(f"\n{report}")

    logger.info("✓ Quality checker test passed!\n")
    return True


def test_dxf_generator(logger, config):
    """Test the DXF generator."""
    logger.info("="*60)
    logger.info("Testing DXF Generator")
    logger.info("="*60)

    generator = DXFGenerator(config['dxf_output'])
    logger.info(f"Generator: {generator}")

    # Create simple test DXF
    logger.info("\n  Creating simple test DXF...")
    output_path = "output/test_simple.dxf"

    try:
        result_path = generator.create_simple_test_dxf(output_path)
        logger.info(f"    ✓ Created: {result_path}")

        # Verify file exists
        assert Path(result_path).exists(), "DXF file should exist"
        file_size = Path(result_path).stat().st_size
        logger.info(f"    File size: {file_size} bytes")

        # Load and validate
        doc = ezdxf.readfile(result_path)
        logger.info(f"    DXF version: {doc.dxfversion}")
        logger.info(f"    Units: {doc.header.get('$INSUNITS', 'not set')}")

        # Get entity counts
        counts = generator.get_entity_count(doc)
        logger.info(f"    Entities: {counts}")

        assert counts['lwpolyline'] > 0, "Should have polylines"
        assert counts['circle'] > 0, "Should have circles"

    except Exception as e:
        logger.error(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with custom contours
    logger.info("\n  Creating DXF from custom contours...")

    # Create contours in millimeters
    rect1 = np.array([[0, 0], [50, 0], [50, 30], [0, 30], [0, 0]], dtype=np.float64)
    rect2 = np.array([[60, 0], [90, 0], [90, 20], [60, 20], [60, 0]], dtype=np.float64)
    triangle = np.array([[100, 0], [125, 0], [112.5, 25], [100, 0]], dtype=np.float64)

    contours = [rect1, rect2, triangle]

    metadata = {
        'SOURCE': 'test_phase3.py',
        'NUM_CONTOURS': str(len(contours))
    }

    output_path2 = "output/test_contours.dxf"

    try:
        result_path = generator.create_dxf(contours, output_path2, metadata=metadata)
        logger.info(f"    ✓ Created: {result_path}")

        # Load and verify
        doc = ezdxf.readfile(result_path)
        counts = generator.get_entity_count(doc)
        logger.info(f"    Entities: {counts}")

        assert counts['lwpolyline'] == len(contours), f"Should have {len(contours)} polylines"

    except Exception as e:
        logger.error(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("✓ DXF generator test passed!\n")
    return True


def main():
    """Run all Phase 3 tests."""
    print("\n" + "="*60)
    print("TESTING PHASE 3 COMPONENTS")
    print("="*60)

    # Setup logger
    logger = setup_logger(name="phase3", level="INFO", console_output=True)

    # Load config
    config = load_config()
    logger.info("Configuration loaded\n")

    all_passed = True

    try:
        # Test entity helpers
        if not test_entity_helpers(logger):
            all_passed = False

        # Test quality checker
        if not test_quality_checker(logger, config):
            all_passed = False

        # Test DXF generator
        if not test_dxf_generator(logger, config):
            all_passed = False

    except Exception as e:
        logger.error(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("="*60)
    if all_passed:
        print("✓ ALL PHASE 3 TESTS PASSED!")
        print("="*60)
        print("\nDXF generation is working correctly!")
        print("\nGenerated files:")
        print("  • output/test_simple.dxf")
        print("  • output/test_contours.dxf")
        print("\nNext: Test complete image → DXF pipeline")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
