#!/usr/bin/env python
"""
Complete end-to-end test: Image → DXF
Tests the complete pipeline from image input to DXF file output.
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
from vectorization.edge_detector import EdgeDetector
from vectorization.contour_extractor import ContourExtractor
from vectorization.path_simplifier import PathSimplifier
from validation.quality_checker import QualityChecker
from dxf.generator import DXFGenerator


def create_test_image():
    """Create a test image for the pipeline."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # Draw simple shapes
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), 2)
    cv2.circle(img, (250, 100), 50, (0, 0, 0), 2)

    pts = np.array([[100, 250], [200, 250], [150, 350]], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)

    return img


def test_complete_pipeline(logger, config):
    """Test the complete image-to-DXF pipeline."""
    logger.info("="*70)
    logger.info("COMPLETE PIPELINE TEST: Image → DXF")
    logger.info("="*70)

    # Step 0: Create test image
    logger.info("\nStep 0: Creating test image...")
    image = create_test_image()
    test_image_path = "output/pipeline_test_input.png"
    cv2.imwrite(test_image_path, image)
    logger.info(f"  ✓ Created: {test_image_path}")
    logger.info(f"  Image shape: {image.shape}")

    # Step 1: Preprocessing
    logger.info("\nStep 1: Preprocessing...")
    preprocessor = DrawingPreprocessor(config['drawing_preprocessing'])
    binary = preprocessor.preprocess(image)
    logger.info(f"  ✓ Preprocessed to binary ({np.unique(binary)})")

    cv2.imwrite("output/pipeline_01_binary.png", binary)

    # Step 2: Edge Detection
    logger.info("\nStep 2: Edge Detection...")
    detector = EdgeDetector(config['edge_detection'])
    edges = detector.detect_edges(binary)
    edge_pixels = np.sum(edges > 0)
    logger.info(f"  ✓ Detected {edge_pixels:,} edge pixels")

    cv2.imwrite("output/pipeline_02_edges.png", edges)

    # Step 3: Contour Extraction
    logger.info("\nStep 3: Contour Extraction...")
    extractor = ContourExtractor(config['contour_extraction'])
    contours_px, hierarchy = extractor.extract_contours(edges)
    total_points = sum(len(cnt) for cnt in contours_px)
    logger.info(f"  ✓ Extracted {len(contours_px)} contours")
    logger.info(f"  Total points: {total_points:,}")

    # Step 4: Path Simplification
    logger.info("\nStep 4: Path Simplification...")
    scale_calc = ScaleCalculator(dpi=96)
    simplifier = PathSimplifier(config['vectorization'], scale_calc)
    simplified_px = simplifier.simplify_contours(contours_px)
    simp_points = sum(len(cnt) for cnt in simplified_px)
    logger.info(f"  ✓ Simplified to {simp_points:,} points")

    # Step 5: Scale to Millimeters
    logger.info("\nStep 5: Scale Conversion (Pixels → Millimeters)...")
    contours_mm = scale_calc.scale_contours(simplified_px)
    logger.info(f"  ✓ Scaled {len(contours_mm)} contours to millimeters")
    logger.info(f"  Scale: {scale_calc}")

    # Show sample dimensions
    if len(contours_mm) > 0:
        sample = contours_mm[0]
        min_x, min_y = sample.min(axis=0)
        max_x, max_y = sample.max(axis=0)
        width_mm = max_x - min_x
        height_mm = max_y - min_y
        logger.info(f"  Sample contour: {width_mm:.2f} x {height_mm:.2f} mm")

    # Step 6: Validation
    logger.info("\nStep 6: Validation...")
    checker = QualityChecker(config['validation'])
    is_valid, warnings, errors = checker.validate_contours(contours_mm)

    logger.info(f"  Valid: {is_valid}")
    logger.info(f"  Warnings: {len(warnings)}")
    logger.info(f"  Errors: {len(errors)}")

    if errors:
        for error in errors[:5]:  # Show first 5
            logger.warning(f"    ERROR: {error.message}")

    if warnings:
        for warning in warnings[:5]:  # Show first 5
            logger.info(f"    WARNING: {warning.message}")

    # Step 7: DXF Generation
    logger.info("\nStep 7: DXF Generation...")
    generator = DXFGenerator(config['dxf_output'])

    metadata = {
        'SOURCE': 'pipeline_test_input.png',
        'NUM_CONTOURS': str(len(contours_mm)),
        'SCALE_DPI': str(scale_calc.dpi),
        'TOTAL_POINTS': str(simp_points)
    }

    output_path = "output/pipeline_test_result.dxf"

    try:
        result_path = generator.create_dxf(contours_mm, output_path, metadata=metadata)
        logger.info(f"  ✓ Generated: {result_path}")

        # Verify DXF file
        import ezdxf
        doc = ezdxf.readfile(result_path)
        counts = generator.get_entity_count(doc)

        logger.info(f"  DXF version: {doc.dxfversion}")
        logger.info(f"  Units: {doc.header.get('$INSUNITS', 'not set')}")
        logger.info(f"  Entities: {counts}")

        file_size = Path(result_path).stat().st_size
        logger.info(f"  File size: {file_size:,} bytes")

        return True

    except Exception as e:
        logger.error(f"  ✗ Failed to generate DXF: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run complete pipeline test."""
    print("\n" + "="*70)
    print("COMPLETE PIPELINE TEST")
    print("="*70)

    # Setup logger
    logger = setup_logger(name="pipeline", level="INFO", console_output=True)

    # Load config
    config = load_config()

    success = test_complete_pipeline(logger, config)

    print("\n" + "="*70)
    if success:
        print("✓ COMPLETE PIPELINE TEST PASSED!")
        print("="*70)
        print("\nGenerated files:")
        print("  Input:  output/pipeline_test_input.png")
        print("  Binary: output/pipeline_01_binary.png")
        print("  Edges:  output/pipeline_02_edges.png")
        print("  DXF:    output/pipeline_test_result.dxf")
        print("\n✓ Image successfully converted to DXF!")
        print("\nYou can now:")
        print("  1. Open the DXF file in LibreCAD or AutoCAD")
        print("  2. Verify the scale is correct (millimeters)")
        print("  3. Use it for laser cutting!")
        return 0
    else:
        print("✗ PIPELINE TEST FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
