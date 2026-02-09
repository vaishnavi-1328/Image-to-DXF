# How to Use: Image to DXF Converter

## ✅ System Status: COMPLETE & WORKING!

Your image-to-DXF conversion system is fully implemented and tested. You can now convert images (photos or line drawings) into DXF files ready for laser cutting.

## Quick Start

### Option 1: Use Existing Test (Easiest)

```bash
cd /Users/vaishnavis/Desktop/img_to_dfx

# Run complete pipeline with sample image
python test_with_real_image.py
```

This will:
- Create a sample drawing
- Process it through the entire pipeline
- Generate DXF file at `output/result.dxf`

### Option 2: Convert Your Own Image

```python
import sys
sys.path.insert(0, 'src')

import cv2
from utils.config_loader import load_config
from utils.scale_calculator import ScaleCalculator
from preprocessing.drawing_preprocessor import DrawingPreprocessor
from vectorization.edge_detector import EdgeDetector
from vectorization.contour_extractor import ContourExtractor
from vectorization.path_simplifier import PathSimplifier
from validation.quality_checker import QualityChecker
from dxf.generator import DXFGenerator

# Load configuration
config = load_config()

# Load your image
image = cv2.imread('path/to/your/image.jpg')

# Step 1: Preprocess
preprocessor = DrawingPreprocessor(config['drawing_preprocessing'])
binary = preprocessor.preprocess(image)

# Step 2: Detect edges
detector = EdgeDetector(config['edge_detection'])
edges = detector.detect_edges(binary)

# Step 3: Extract contours
extractor = ContourExtractor(config['contour_extraction'])
contours_px, _ = extractor.extract_contours(edges)

# Step 4: Simplify paths
scale_calc = ScaleCalculator(dpi=96)  # Adjust DPI as needed
simplifier = PathSimplifier(config['vectorization'], scale_calc)
simplified_px = simplifier.simplify_contours(contours_px)

# Step 5: Scale to millimeters
contours_mm = scale_calc.scale_contours(simplified_px)

# Step 6: Validate
checker = QualityChecker(config['validation'])
is_valid, warnings, errors = checker.validate_contours(contours_mm)

if errors:
    print("Errors found:")
    for error in errors:
        print(f"  • {error.message}")
else:
    print("Validation passed!")

# Step 7: Generate DXF
generator = DXFGenerator(config['dxf_output'])
output_path = generator.create_dxf(contours_mm, "output/my_result.dxf")

print(f"✓ DXF file created: {output_path}")
```

## Testing the System

### Run All Tests

```bash
# Test foundation (config, logging, scaling)
python test_foundation.py

# Test preprocessing & vectorization
python test_phase2.py

# Test DXF generation
python test_phase3.py

# Test complete pipeline
python test_complete_pipeline.py
```

## Important Settings

### Scale/DPI Settings

The scale determines the real-world size of your output. Adjust in `ScaleCalculator`:

```python
# If you know the DPI of your image
scale_calc = ScaleCalculator(dpi=300)  # Higher DPI = larger output

# If you know a reference dimension
scale_calc = ScaleCalculator(
    reference_pixels=1000,  # A line is 1000 pixels long
    reference_mm=100        # That line should be 100mm in real life
)

# Direct pixels per millimeter
scale_calc = ScaleCalculator(pixels_per_mm=10)  # 10 pixels = 1mm
```

### Processing Parameters

Edit `config/defaults.yaml` to adjust:

```yaml
# For noisy images
photo_preprocessing:
  bilateral_filter:
    d: 11  # Increase for more noise removal

# For more/fewer edges
edge_detection:
  canny_low_threshold: 30   # Lower = more edges
  canny_high_threshold: 100  # Higher = fewer edges

# For smaller features
contour_extraction:
  min_contour_area_pixels: 50  # Lower = keep smaller features

# For simpler paths
vectorization:
  simplify_epsilon_mm: 0.1  # Higher = fewer points, simpler paths
```

## Output Files

After running the pipeline, you'll find:

- **01_binary.png** - Preprocessed binary image
- **02_edges.png** - Detected edges
- **02_edges_overlay.png** - Edges on original image
- **03_contours.png** - Extracted vector contours
- **04_simplified.png** - Simplified paths
- **result.dxf** - Final DXF file for laser cutting

## DXF File Specifications

The generated DXF files are optimized for laser cutting:

- **Format**: R2010 (compatible with most CAD software)
- **Units**: Millimeters at 1:1 scale
- **Layer**: "CutLines" (red color)
- **Entity Type**: LWPolylines (closed paths)
- **Coordinate System**: 2D (Z=0)

## Opening DXF Files

You can open the generated DXF files in:

- **LibreCAD** (free, open source)
- **AutoCAD** (commercial)
- **QCAD** (free/commercial)
- **DraftSight** (free/commercial)
- **Any laser cutting software** that accepts DXF

## Troubleshooting

### Problem: DXF file scale is wrong

**Solution**: Adjust the DPI or provide a reference dimension:

```python
# Instead of dpi=96, use your actual DPI
scale_calc = ScaleCalculator(dpi=300)

# Or provide a known dimension
scale_calc = ScaleCalculator(
    reference_pixels=500,  # This feature is 500 pixels
    reference_mm=50        # It should be 50mm in reality
)
```

### Problem: Too many small contours

**Solution**: Increase `min_contour_area_pixels`:

```python
# In code
config['contour_extraction']['min_contour_area_pixels'] = 200

# Or edit config/defaults.yaml:
contour_extraction:
  min_contour_area_pixels: 200
```

### Problem: Paths are not closed

**Solution**: The system automatically closes paths. If validation reports open paths:
1. Check if the gap is small (<1mm) - this will be auto-closed
2. Increase morphological closing iterations in preprocessing
3. Adjust edge detection thresholds

### Problem: Important features missing

**Solution**:
1. Lower Canny thresholds: `canny_low_threshold: 30`
2. Decrease `min_contour_area_pixels`
3. Try photo preprocessing instead of drawing preprocessing

### Problem: Validation errors

**Solution**: The system reports specific errors. Common fixes:
- **Self-intersecting paths**: Usually false positives from the simplified check - safe to ignore for simple shapes
- **Too few points**: Feature is too small, increase minimum size or filter it out
- **Not closed**: Small gap - will be auto-closed during DXF generation

## Example Workflow

### For a Photo of a Gate:

```bash
python test_with_real_image.py gate_photo.jpg photo
```

Settings to adjust:
- DPI: 300 (if high-resolution photo)
- Preprocessing: photo (adaptive thresholding)
- Simplification: 0.05mm (default is good)

### For a CAD Drawing:

```bash
python test_with_real_image.py drawing.png drawing
```

Settings:
- DPI: Based on export resolution
- Preprocessing: drawing (Otsu thresholding)
- Simplification: 0.02mm (tighter for precision)

## What's Included

### Implemented Modules:

1. **Foundation** (Phase 1)
   - Configuration loader (YAML + CLI)
   - Logging system
   - Scale calculator

2. **Preprocessing** (Phase 2)
   - Drawing preprocessor (Otsu + median filter)
   - Photo preprocessor (bilateral + CLAHE + adaptive)
   - Base preprocessor (utilities)

3. **Vectorization** (Phase 2)
   - Edge detector (Canny)
   - Contour extractor (OpenCV with hierarchy)
   - Path simplifier (Douglas-Peucker + circle detection)

4. **DXF Generation** (Phase 3)
   - Entity helpers (LWPolyline, Circle, Arc)
   - DXF generator (R2010 format)
   - Quality checker (validation)

### Test Scripts:

- `test_foundation.py` - Tests config, logging, scale
- `test_phase2.py` - Tests preprocessing & vectorization
- `test_phase3.py` - Tests DXF generation & validation
- `test_complete_pipeline.py` - End-to-end image → DXF
- `test_with_real_image.py` - Process real images with previews
- `quick_test.py` - One-command quick test

## Next Steps

1. **Test with your images**: Run `python test_with_real_image.py your_image.jpg`
2. **Verify DXF output**: Open generated DXF files in LibreCAD or AutoCAD
3. **Adjust settings**: Edit `config/defaults.yaml` to tune for your images
4. **Send to laser cutter**: Use the DXF files for laser cutting metal sheets!

## Success!

Your complete image-to-DXF conversion system is ready to use. All components have been tested and are working correctly. You now have:

✓ Image preprocessing (photos & drawings)
✓ Edge detection and vectorization
✓ Path simplification
✓ Quality validation
✓ DXF file generation (laser-ready)
✓ Complete testing suite
✓ Comprehensive documentation

Happy laser cutting! 🎉
