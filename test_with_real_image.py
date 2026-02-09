"""
Test the complete pipeline with the test_tilted.jpg image
This simulates what happens in the Streamlit app
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing.perspective_corrector import PerspectiveCorrector
from preprocessing.drawing_preprocessor import DrawingPreprocessor
from vectorization.edge_detector import EdgeDetector
from vectorization.contour_extractor import ContourExtractor
from utils.config_loader import load_config

# Load config
config = load_config()

# Load the tilted test image
print("Loading test_tilted.jpg...")
tilted = cv2.imread("test_tilted.jpg")
print(f"Image size: {tilted.shape[1]}x{tilted.shape[0]}")

# The corners we know from the test script
corners = [
    (150, 120),  # Top-left
    (620, 180),  # Top-right
    (580, 470),  # Bottom-right
    (180, 430)   # Bottom-left
]

print("\n=== STEP 1: Apply Perspective Correction ===")
corrector = PerspectiveCorrector(tilted, config.get('perspective_correction', {}))
corrector.set_source_points(corners)
corrector.set_output_dimensions(400.0, 300.0)

corrected = corrector.apply_correction()
dpi = corrector.get_calculated_dpi()

print(f"Corrected image size: {corrected.shape[1]}x{corrected.shape[0]}")
print(f"Calculated DPI: {dpi:.1f}")

# Save corrected for inspection
cv2.imwrite("debug_corrected.jpg", corrected)
print("Saved debug_corrected.jpg")

print("\n=== STEP 2: Preprocess (as Drawing) ===")
preprocessor = DrawingPreprocessor(config['drawing_preprocessing'])
binary = preprocessor.preprocess(corrected)
cv2.imwrite("debug_binary.jpg", binary)
print("Saved debug_binary.jpg")

print("\n=== STEP 3: Edge Detection ===")
detector = EdgeDetector(config['edge_detection'])
edges = detector.detect_edges(binary)
cv2.imwrite("debug_edges.jpg", edges)
print(f"Edge pixels detected: {np.sum(edges > 0)}")
print("Saved debug_edges.jpg")

print("\n=== STEP 4: Contour Extraction ===")
extractor = ContourExtractor(config['contour_extraction'])
contours_px, hierarchy = extractor.extract_contours(edges)
print(f"Contours found: {len(contours_px)}")

if len(contours_px) > 0:
    print(f"Total points: {sum(len(c) for c in contours_px)}")

    # Visualize contours
    contour_vis = extractor.visualize_contours(corrected, contours_px)
    cv2.imwrite("debug_contours.jpg", contour_vis)
    print("Saved debug_contours.jpg")

    print("\n✅ Pipeline successful!")
else:
    print("\n⚠️ No contours found!")
    print("This might be why you're seeing validation errors.")

print("\n=== Files to check ===")
print("1. debug_corrected.jpg - Should show straightened rectangle")
print("2. debug_binary.jpg - Should show black & white")
print("3. debug_edges.jpg - Should show white edges on black")
print("4. debug_contours.jpg - Should show detected contours")
