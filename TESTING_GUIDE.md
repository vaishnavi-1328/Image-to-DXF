# Testing Guide - Image to DXF Converter

## Quick Start - Test Everything at Once

### 1. Run the Complete Test Suite

```bash
cd /Users/vaishnavis/Desktop/img_to_dfx

# Test foundation components (config, logging, scaling)
python test_foundation.py

# Test Phase 2 components (preprocessing, vectorization)
python test_phase2.py

# Test with a real image (full pipeline)
python test_with_real_image.py
```

### 2. Test with Your Own Image

```bash
# For line drawings
python test_with_real_image.py path/to/your/drawing.png

# For photos
python test_with_real_image.py path/to/your/photo.jpg photo
```

### 3. View the Results

After running `test_with_real_image.py`, check the `output/` folder:

- **01_binary.png** - Preprocessed binary image (black and white)
- **02_edges.png** - Detected edges
- **02_edges_overlay.png** - Edges overlaid on original (green lines)
- **03_contours.png** - Extracted contours (vector paths)
- **04_simplified.png** - Simplified paths (fewer points, same shape)

## Testing Individual Components

### Test Preprocessing Only

```python
import sys
sys.path.insert(0, 'src')

import cv2
from utils.config_loader import load_config
from preprocessing.drawing_preprocessor import DrawingPreprocessor

# Load config
config = load_config()

# Load your image
image = cv2.imread('path/to/image.jpg')

# Preprocess
preprocessor = DrawingPreprocessor(config['drawing_preprocessing'])
binary = preprocessor.preprocess(image)

# Save result
cv2.imwrite('output/test_binary.png', binary)
print("Saved: output/test_binary.png")
```

### Test Edge Detection Only

```python
import sys
sys.path.insert(0, 'src')

import cv2
from utils.config_loader import load_config
from vectorization.edge_detector import EdgeDetector

# Load config
config = load_config()

# Load preprocessed image (must be binary)
binary = cv2.imread('output/01_binary.png', cv2.IMREAD_GRAYSCALE)

# Detect edges
detector = EdgeDetector(config['edge_detection'])
edges = detector.detect_edges(binary)

# Save result
cv2.imwrite('output/test_edges.png', edges)
print(f"Detected {edges.sum() // 255} edge pixels")
```

### Test Contour Extraction Only

```python
import sys
sys.path.insert(0, 'src')

import cv2
from utils.config_loader import load_config
from vectorization.contour_extractor import ContourExtractor

# Load config
config = load_config()

# Load edge map
edges = cv2.imread('output/02_edges.png', cv2.IMREAD_GRAYSCALE)

# Extract contours
extractor = ContourExtractor(config['contour_extraction'])
contours, hierarchy = extractor.extract_contours(edges)

print(f"Extracted {len(contours)} contours")
for i, cnt in enumerate(contours[:5]):
    props = extractor.get_contour_properties(cnt)
    print(f"  Contour {i}: {props['area']:.1f} px², {props['num_points']} points")
```

### Test Path Simplification Only

```python
import sys
sys.path.insert(0, 'src')

from utils.config_loader import load_config
from utils.scale_calculator import ScaleCalculator
from vectorization.path_simplifier import PathSimplifier

# Load config
config = load_config()

# Assume you have contours from previous step
# contours = [...]

# Create scale calculator
scale_calc = ScaleCalculator(dpi=96)

# Simplify paths
simplifier = PathSimplifier(config['vectorization'], scale_calc)
simplified = simplifier.simplify_contours(contours)

orig_points = sum(len(cnt) for cnt in contours)
simp_points = sum(len(cnt) for cnt in simplified)

print(f"Original: {orig_points} points")
print(f"Simplified: {simp_points} points")
print(f"Reduction: {(orig_points - simp_points) / orig_points * 100:.1f}%")
```

## Common Use Cases

### Case 1: Process a Simple Line Drawing

```bash
# Create or use an existing line drawing
python test_with_real_image.py my_drawing.png drawing
```

**Best for:**
- CAD exports
- Vector graphics
- Clean sketches
- High contrast images

### Case 2: Process a Photo

```bash
python test_with_real_image.py my_photo.jpg photo
```

**Best for:**
- Photos of gates, elevations
- Images with shadows/lighting variation
- Noisy images

### Case 3: Adjust Processing Parameters

Edit `config/defaults.yaml` to tune the processing:

```yaml
# For noisier images, increase filtering
photo_preprocessing:
  bilateral_filter:
    d: 11  # Increase from 9

# For more aggressive edge detection
edge_detection:
  canny_low_threshold: 30  # Decrease from 50
  canny_high_threshold: 100  # Decrease from 150

# To capture smaller features
contour_extraction:
  min_contour_area_pixels: 50  # Decrease from 100

# For more aggressive simplification
vectorization:
  simplify_epsilon_mm: 0.1  # Increase from 0.05
```

## Understanding the Output

### Preprocessing Output (01_binary.png)
- Pure black and white image
- Black = features to trace
- White = background
- If too noisy: increase filtering
- If features missing: adjust threshold

### Edge Detection Output (02_edges.png)
- White lines on black background
- Should outline all features clearly
- If edges broken: decrease Canny thresholds
- If too many edges: increase thresholds

### Contours Output (03_contours.png)
- Green lines show extracted paths
- Each closed path = one contour
- Count shown in console output
- If too few: decrease min_contour_area
- If too many: increase min_contour_area

### Simplified Output (04_simplified.png)
- Blue lines show simplified paths
- Fewer points, same shape
- Ready for DXF conversion
- Point count shown in console

## Troubleshooting

### Problem: No contours detected
**Solution:**
- Check 01_binary.png - is the image properly binarized?
- Check 02_edges.png - are edges detected?
- Lower `min_contour_area_pixels` in config

### Problem: Too many small contours
**Solution:**
- Increase `min_contour_area_pixels` in config
- Increase morphological closing iterations

### Problem: Edges are broken/fragmented
**Solution:**
- Increase Gaussian blur sigma
- Increase morphological closing iterations
- Decrease Canny thresholds

### Problem: Important features missing
**Solution:**
- Check preprocessing settings
- For drawings: try manual threshold instead of Otsu
- For photos: adjust CLAHE clip_limit
- Decrease Canny thresholds

### Problem: Image dimensions wrong
**Solution:**
- Adjust DPI in ScaleCalculator
- Provide reference dimension: `ScaleCalculator(reference_pixels=1000, reference_mm=100)`

## Performance Tips

### For Large Images (> 4000x4000)
- Resize before processing
- Increase min_contour_area to filter small features
- Increase simplification epsilon

### For Complex Patterns
- Use lower simplification epsilon (0.01-0.03 mm)
- Enable circle detection for round features
- Process in sections if too slow

## Next Steps

Once you're happy with the contour extraction:

1. **Validate contours** - Check that all paths are closed
2. **Generate DXF** - Convert contours to DXF format (Phase 3)
3. **Test in CAD** - Open DXF in LibreCAD or AutoCAD
4. **Send to laser** - Use with your laser cutting machine

## What's Been Tested

✓ Configuration loading
✓ Logging system
✓ Scale calculations
✓ Drawing preprocessing
✓ Photo preprocessing
✓ Edge detection
✓ Contour extraction (with hierarchy)
✓ Path simplification
✓ Circle detection
✓ Full end-to-end pipeline

## What's Next (Phase 3)

- DXF file generation
- Path validation
- Quality checking
- Preview system
- CLI interface
