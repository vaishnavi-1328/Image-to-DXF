"""
Manual test script for perspective correction
This creates a tilted test image and corrects it to verify the implementation works
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing.perspective_corrector import PerspectiveCorrector

# Create a test image with a white rectangle
print("Creating test image with rectangle...")
original = np.zeros((600, 800, 3), dtype=np.uint8)
cv2.rectangle(original, (200, 150), (600, 450), (255, 255, 255), -1)
cv2.putText(original, "ORIGINAL", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

# Define perfect rectangle corners
src_pts = np.float32([[200, 150], [600, 150], [600, 450], [200, 450]])

# Create tilted perspective (simulating angled photo)
dst_pts = np.float32([
    [150, 120],  # Top-left (shifted)
    [620, 180],  # Top-right (shifted)
    [580, 470],  # Bottom-right (shifted)
    [180, 430]   # Bottom-left (shifted)
])

# Apply perspective transformation to create tilted image
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
tilted = cv2.warpPerspective(original, matrix, (800, 600))

# Add label to tilted image
cv2.putText(tilted, "TILTED", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Save tilted image
cv2.imwrite("test_tilted.jpg", tilted)
print("Saved test_tilted.jpg")

# Now use PerspectiveCorrector to correct it back
print("\nApplying perspective correction...")
corrector = PerspectiveCorrector(tilted)

# Use the tilted corners as source points
corners = [
    (150, 120),  # Top-left
    (620, 180),  # Top-right
    (580, 470),  # Bottom-right
    (180, 430)   # Bottom-left
]

corrector.set_source_points(corners)
corrector.set_output_dimensions(400.0, 300.0)  # Real dimensions in mm

# Apply correction
corrected = corrector.apply_correction()
dpi = corrector.get_calculated_dpi()

print(f"Original size: {original.shape[1]}x{original.shape[0]}")
print(f"Tilted size: {tilted.shape[1]}x{tilted.shape[0]}")
print(f"Corrected size: {corrected.shape[1]}x{corrected.shape[0]}")
print(f"Calculated DPI: {dpi:.1f}")

# Save corrected image
cv2.imwrite("test_corrected.jpg", corrected)
print("\nSaved test_corrected.jpg")

print("\n✅ Test complete!")
print("Check the files:")
print("  - test_tilted.jpg (the input image with perspective distortion)")
print("  - test_corrected.jpg (should be straightened)")
print("\nThe corrected image should show a straight rectangle, not tilted.")
