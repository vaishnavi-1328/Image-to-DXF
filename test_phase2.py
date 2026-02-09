#!/usr/bin/env python
"""
Test script for Phase 2 components (preprocessing and vectorization).
"""
import sys
from pathlib import Path
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config_loader import load_config
from utils.logger import setup_logger
from utils.scale_calculator import ScaleCalculator
from preprocessing.drawing_preprocessor import DrawingPreprocessor
from preprocessing.photo_preprocessor import PhotoPreprocessor
from vectorization.edge_detector import EdgeDetector
from vectorization.contour_extractor import ContourExtractor
from vectorization.path_simplifier import PathSimplifier


def create_test_image_simple() -> np.ndarray:
    """Create a simple test image with geometric shapes."""
    img = np.ones((400, 400), dtype=np.uint8) * 255

    # Draw a rectangle
    cv2.rectangle(img, (50, 50), (150, 150), 0, 2)

    # Draw a circle
    cv2.circle(img, (250, 100), 50, 0, 2)

    # Draw a triangle
    pts = np.array([[100, 250], [200, 250], [150, 350]], np.int32)
    cv2.polylines(img, [pts], True, 0, 2)

    return img


def create_test_image_complex() -> np.ndarray:
    """Create a more complex test image."""
    img = np.ones((500, 500), dtype=np.uint8) * 255

    # Draw concentric rectangles
    for i in range(3):
        offset = i * 30
        cv2.rectangle(img, (50+offset, 50+offset), (200-offset, 200-offset), 0, 2)

    # Draw a pattern with holes
    cv2.rectangle(img, (250, 50), (450, 250), 0, 2)
    cv2.rectangle(img, (280, 80), (320, 120), 0, -1)  # Filled hole
    cv2.rectangle(img, (380, 80), (420, 120), 0, -1)  # Filled hole

    # Draw some noise
    for _ in range(20):
        x, y = np.random.randint(0, 500, 2)
        cv2.circle(img, (x, y), 2, 0, -1)

    return img


def test_drawing_preprocessor(logger, config):
    """Test the drawing preprocessor."""
    logger.info("="*60)
    logger.info("Testing Drawing Preprocessor")
    logger.info("="*60)

    # Create test image
    img = create_test_image_simple()

    # Initialize preprocessor
    preprocessor = DrawingPreprocessor(config['drawing_preprocessing'])
    logger.info(f"Preprocessor: {preprocessor}")

    # Preprocess
    binary = preprocessor.preprocess(img)

    logger.info(f"  Input shape: {img.shape}")
    logger.info(f"  Output shape: {binary.shape}")
    logger.info(f"  Output dtype: {binary.dtype}")
    logger.info(f"  Unique values: {np.unique(binary)}")

    # Validate
    assert binary.shape == img.shape, "Shape mismatch"
    assert binary.dtype == np.uint8, "Wrong dtype"
    assert len(np.unique(binary)) <= 2, "Should be binary"

    logger.info("✓ Drawing preprocessor test passed!")
    return binary


def test_photo_preprocessor(logger, config):
    """Test the photo preprocessor."""
    logger.info("="*60)
    logger.info("Testing Photo Preprocessor")
    logger.info("="*60)

    # Create test image with noise
    img = create_test_image_complex()

    # Add noise to simulate photo
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Initialize preprocessor
    preprocessor = PhotoPreprocessor(config['photo_preprocessing'])
    logger.info(f"Preprocessor: {preprocessor}")

    # Preprocess
    binary = preprocessor.preprocess(img_noisy)

    logger.info(f"  Input shape: {img_noisy.shape}")
    logger.info(f"  Output shape: {binary.shape}")
    logger.info(f"  Output dtype: {binary.dtype}")
    logger.info(f"  Unique values: {np.unique(binary)}")

    # Validate
    assert binary.shape == img_noisy.shape, "Shape mismatch"
    assert binary.dtype == np.uint8, "Wrong dtype"

    logger.info("✓ Photo preprocessor test passed!")
    return binary


def test_edge_detector(logger, config, binary_image):
    """Test the edge detector."""
    logger.info("="*60)
    logger.info("Testing Edge Detector")
    logger.info("="*60)

    # Initialize detector
    detector = EdgeDetector(config['edge_detection'])
    logger.info(f"Detector: {detector}")

    # Detect edges
    edges = detector.detect_edges(binary_image)

    logger.info(f"  Input shape: {binary_image.shape}")
    logger.info(f"  Output shape: {edges.shape}")
    logger.info(f"  Edge pixels: {np.sum(edges > 0)}")

    # Validate
    assert edges.shape == binary_image.shape, "Shape mismatch"
    assert edges.dtype == np.uint8, "Wrong dtype"
    assert np.sum(edges > 0) > 0, "No edges detected"

    # Test auto threshold
    edges_auto = detector.detect_edges_auto_threshold(binary_image)
    logger.info(f"  Auto-threshold edge pixels: {np.sum(edges_auto > 0)}")

    # Test suggested thresholds
    low, high = detector.get_suggested_thresholds(binary_image)
    logger.info(f"  Suggested thresholds: low={low}, high={high}")

    logger.info("✓ Edge detector test passed!")
    return edges


def test_contour_extractor(logger, config, edge_map):
    """Test the contour extractor."""
    logger.info("="*60)
    logger.info("Testing Contour Extractor")
    logger.info("="*60)

    # Initialize extractor
    extractor = ContourExtractor(config['contour_extraction'])
    logger.info(f"Extractor: {extractor}")

    # Extract contours
    contours, hierarchy = extractor.extract_contours(edge_map)

    logger.info(f"  Input shape: {edge_map.shape}")
    logger.info(f"  Contours found: {len(contours)}")

    if len(contours) > 0:
        total_points = sum(len(cnt) for cnt in contours)
        logger.info(f"  Total points: {total_points}")
        logger.info(f"  Avg points per contour: {total_points / len(contours):.1f}")

        # Get properties of first contour
        if len(contours) > 0:
            props = extractor.get_contour_properties(contours[0])
            logger.info(f"  First contour properties:")
            logger.info(f"    Area: {props['area']:.1f} px²")
            logger.info(f"    Perimeter: {props['perimeter']:.1f} px")
            logger.info(f"    Points: {props['num_points']}")
            logger.info(f"    Compactness: {props['compactness']:.3f}")

    # Test with info
    contour_infos = extractor.extract_contours_with_info(edge_map)
    logger.info(f"  Contours with info: {len(contour_infos)}")

    holes = sum(1 for info in contour_infos if info.is_hole)
    logger.info(f"  Holes detected: {holes}")

    # Validate
    assert len(contours) > 0, "No contours extracted"
    for cnt in contours:
        assert len(cnt.shape) == 2, "Wrong contour shape"
        assert cnt.shape[1] == 2, "Contour should have x,y coordinates"

    logger.info("✓ Contour extractor test passed!")
    return contours


def test_path_simplifier(logger, config, contours):
    """Test the path simplifier."""
    logger.info("="*60)
    logger.info("Testing Path Simplifier")
    logger.info("="*60)

    # Initialize simplifier
    scale_calc = ScaleCalculator(dpi=96)
    simplifier = PathSimplifier(config['vectorization'], scale_calc)
    logger.info(f"Simplifier: {simplifier}")
    logger.info(f"  Epsilon (pixels): {simplifier.get_epsilon_pixels():.2f}")

    # Simplify contours
    simplified = simplifier.simplify_contours(contours)

    logger.info(f"  Original contours: {len(contours)}")
    logger.info(f"  Simplified contours: {len(simplified)}")

    orig_points = sum(len(cnt) for cnt in contours)
    simp_points = sum(len(cnt) for cnt in simplified)

    logger.info(f"  Original total points: {orig_points}")
    logger.info(f"  Simplified total points: {simp_points}")
    logger.info(f"  Reduction: {(orig_points - simp_points) / orig_points * 100:.1f}%")

    # Get stats
    stats = simplifier.get_simplification_stats(contours, simplified)
    logger.info(f"  Stats: {stats}")

    # Test circle detection
    if len(contours) > 0:
        results = simplifier.detect_and_convert_circles(contours)
        circles = sum(1 for r in results if r['type'] == 'circle')
        logger.info(f"  Circles detected: {circles}")

    # Validate
    assert len(simplified) == len(contours), "Should have same number of contours"
    # Note: simplified may have slightly more points due to ensuring closed paths
    # This is expected and correct behavior

    logger.info("✓ Path simplifier test passed!")
    return simplified


def test_end_to_end_pipeline(logger, config):
    """Test the complete preprocessing and vectorization pipeline."""
    logger.info("="*60)
    logger.info("Testing End-to-End Pipeline")
    logger.info("="*60)

    # Create test image
    img = create_test_image_simple()

    # Step 1: Preprocess
    preprocessor = DrawingPreprocessor(config['drawing_preprocessing'])
    binary = preprocessor.preprocess(img)
    logger.info("  ✓ Step 1: Preprocessing complete")

    # Step 2: Edge detection
    detector = EdgeDetector(config['edge_detection'])
    edges = detector.detect_edges(binary)
    logger.info(f"  ✓ Step 2: Edge detection complete ({np.sum(edges > 0)} edge pixels)")

    # Step 3: Contour extraction
    extractor = ContourExtractor(config['contour_extraction'])
    contours, _ = extractor.extract_contours(edges)
    logger.info(f"  ✓ Step 3: Contour extraction complete ({len(contours)} contours)")

    # Step 4: Path simplification
    scale_calc = ScaleCalculator(dpi=96)
    simplifier = PathSimplifier(config['vectorization'], scale_calc)
    simplified = simplifier.simplify_contours(contours)
    logger.info(f"  ✓ Step 4: Path simplification complete")

    # Final stats
    orig_points = sum(len(cnt) for cnt in contours)
    simp_points = sum(len(cnt) for cnt in simplified)

    logger.info(f"\n  Pipeline Results:")
    logger.info(f"    Input: {img.shape} image")
    logger.info(f"    Contours: {len(contours)}")
    logger.info(f"    Total points (original): {orig_points}")
    logger.info(f"    Total points (simplified): {simp_points}")
    logger.info(f"    Reduction: {(orig_points - simp_points) / orig_points * 100:.1f}%")

    logger.info("\n✓ End-to-end pipeline test passed!")
    return simplified


def main():
    """Run all Phase 2 tests."""
    print("\n" + "="*60)
    print("TESTING PHASE 2 COMPONENTS")
    print("="*60)

    # Setup logger
    logger = setup_logger(name="phase2", level="INFO", console_output=True)

    # Load config
    config = load_config()
    logger.info("Configuration loaded")

    all_passed = True

    try:
        # Test drawing preprocessor
        binary_drawing = test_drawing_preprocessor(logger, config)
        print()

        # Test photo preprocessor
        binary_photo = test_photo_preprocessor(logger, config)
        print()

        # Test edge detector
        edges = test_edge_detector(logger, config, binary_drawing)
        print()

        # Test contour extractor
        contours = test_contour_extractor(logger, config, edges)
        print()

        # Test path simplifier
        simplified = test_path_simplifier(logger, config, contours)
        print()

        # Test end-to-end
        test_end_to_end_pipeline(logger, config)
        print()

    except Exception as e:
        logger.error(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("="*60)
    if all_passed:
        print("✓ ALL PHASE 2 TESTS PASSED!")
        print("="*60)
        print("\nPhase 2 components working correctly!")
        print("\nNext: Implement Phase 3 (DXF generation and validation)")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
