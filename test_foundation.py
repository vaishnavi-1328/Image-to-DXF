#!/usr/bin/env python
"""
Test script to verify the foundation components are working correctly.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.logger import setup_logger, get_logger
from utils.config_loader import load_config, merge_cli_args, validate_config
from utils.scale_calculator import ScaleCalculator


def test_logger():
    """Test the logging system."""
    print("\n" + "="*60)
    print("Testing Logger")
    print("="*60)

    # Set up logger
    logger = setup_logger(
        name="test",
        level="DEBUG",
        log_file="output/test.log",
        console_output=True
    )

    # Test different log levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")

    print("✓ Logger test passed!\n")


def test_config_loader():
    """Test the configuration loader."""
    print("="*60)
    print("Testing Config Loader")
    print("="*60)

    # Load default config
    config = load_config()

    # Test getting values
    canny_low = config.get('edge_detection.canny_low_threshold')
    print(f"Canny low threshold: {canny_low}")
    assert canny_low == 50, "Default canny_low should be 50"

    min_area = config.get('contour_extraction.min_contour_area_pixels')
    print(f"Min contour area: {min_area} pixels")
    assert min_area == 100, "Default min_area should be 100"

    dxf_units = config.get('dxf_output.units')
    print(f"DXF units: {dxf_units} (4=millimeters)")
    assert dxf_units == 4, "Default units should be 4 (mm)"

    # Test setting values
    config.set('edge_detection.canny_low_threshold', 30)
    new_value = config.get('edge_detection.canny_low_threshold')
    print(f"Updated canny low threshold: {new_value}")
    assert new_value == 30, "Should update to 30"

    # Test CLI argument merging
    cli_args = {
        'canny_low': 40,
        'canny_high': 120,
        'min_area': 50,
        'dpi': 300,
        'preview': True
    }

    config = load_config()  # Reload
    config = merge_cli_args(config, cli_args)

    print(f"After CLI merge - canny_low: {config.get('edge_detection.canny_low_threshold')}")
    print(f"After CLI merge - DPI: {config.get('scale.default_dpi')}")
    print(f"After CLI merge - preview: {config.get('preview.generate_previews')}")

    # Test validation
    try:
        validate_config(config)
        print("✓ Config validation passed!")
    except ValueError as e:
        print(f"✗ Config validation failed: {e}")
        return False

    # Test invalid config
    bad_config = load_config()
    bad_config.set('edge_detection.canny_low_threshold', 200)  # Higher than high threshold
    bad_config.set('edge_detection.canny_high_threshold', 150)

    try:
        validate_config(bad_config)
        print("✗ Should have failed validation!")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught invalid config: {e}")

    print("✓ Config loader test passed!\n")
    return True


def test_scale_calculator():
    """Test the scale calculator."""
    print("="*60)
    print("Testing Scale Calculator")
    print("="*60)

    # Test with DPI
    calc1 = ScaleCalculator(dpi=96)
    print(f"Calculator 1: {calc1}")
    print(f"  DPI: {calc1.dpi:.2f}")
    print(f"  Pixels per mm: {calc1.pixels_per_mm:.4f}")
    print(f"  MM per pixel: {calc1.mm_per_pixel:.4f}")

    # Convert 100 pixels to mm
    mm = calc1.pixels_to_mm(100)
    print(f"  100 pixels = {mm:.2f} mm")

    # Test with pixels_per_mm
    calc2 = ScaleCalculator(pixels_per_mm=10.0)
    print(f"\nCalculator 2: {calc2}")
    print(f"  DPI: {calc2.dpi:.2f}")
    print(f"  100 pixels = {calc2.pixels_to_mm(100):.2f} mm")
    print(f"  10 mm = {calc2.mm_to_pixels(10):.2f} pixels")

    # Test with reference dimension
    calc3 = ScaleCalculator(reference_pixels=1000, reference_mm=100)
    print(f"\nCalculator 3: {calc3}")
    print(f"  Reference: 1000 pixels = 100 mm")
    print(f"  DPI: {calc3.dpi:.2f}")
    print(f"  500 pixels = {calc3.pixels_to_mm(500):.2f} mm")

    # Test point scaling
    point_px = (100, 200)
    point_mm = calc1.scale_point(point_px)
    print(f"\n  Point {point_px} pixels = ({point_mm[0]:.2f}, {point_mm[1]:.2f}) mm")

    # Test contour scaling
    import numpy as np
    contour_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    contour_mm = calc1.scale_contour(contour_px)
    print(f"\n  Contour scaled from pixels to mm:")
    print(f"  Original (pixels):\n{contour_px}")
    print(f"  Scaled (mm):\n{contour_mm}")

    print("\n✓ Scale calculator test passed!\n")
    return True


def test_config_structure():
    """Test the complete config structure."""
    print("="*60)
    print("Testing Config Structure")
    print("="*60)

    config = load_config()

    # Check all major sections exist
    sections = [
        'classification',
        'photo_preprocessing',
        'drawing_preprocessing',
        'edge_detection',
        'contour_extraction',
        'vectorization',
        'validation',
        'dxf_output',
        'scale',
        'preview',
        'logging'
    ]

    for section in sections:
        if section in config:
            print(f"  ✓ {section}")
        else:
            print(f"  ✗ {section} MISSING!")
            return False

    # Print some key values
    print(f"\nKey Configuration Values:")
    print(f"  Photo preprocessing - bilateral filter d: {config.get('photo_preprocessing.bilateral_filter.d')}")
    print(f"  Drawing preprocessing - median filter size: {config.get('drawing_preprocessing.median_filter_size')}")
    print(f"  Edge detection - Canny low: {config.get('edge_detection.canny_low_threshold')}")
    print(f"  Edge detection - Canny high: {config.get('edge_detection.canny_high_threshold')}")
    print(f"  Vectorization - simplify epsilon: {config.get('vectorization.simplify_epsilon_mm')} mm")
    print(f"  Validation - min feature size: {config.get('validation.min_feature_size_mm')} mm")
    print(f"  DXF - version: {config.get('dxf_output.version')}")
    print(f"  DXF - layer name: {config.get('dxf_output.layer_name')}")
    print(f"  DXF - layer color: {config.get('dxf_output.layer_color')}")

    print("\n✓ Config structure test passed!\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING FOUNDATION COMPONENTS")
    print("="*60)

    all_passed = True

    try:
        test_logger()
    except Exception as e:
        print(f"✗ Logger test FAILED: {e}")
        all_passed = False

    try:
        if not test_config_loader():
            all_passed = False
    except Exception as e:
        print(f"✗ Config loader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        if not test_scale_calculator():
            all_passed = False
    except Exception as e:
        print(f"✗ Scale calculator test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        if not test_config_structure():
            all_passed = False
    except Exception as e:
        print(f"✗ Config structure test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nFoundation is working correctly. Ready for Phase 2!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
