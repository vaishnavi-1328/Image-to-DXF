# Image-to-DXF Converter for Laser Cutting

Convert images (photos and line drawings) of gates, building elevations, decorative patterns, and other objects into clean DXF files optimized for laser cutting metal sheets.

## Features

- **Adaptive Processing**: Automatically detects and processes photos vs line drawings with optimized algorithms
- **Semi-Automated Workflow**: Preview results and adjust parameters before final DXF generation
- **Laser-Ready Output**: Generates DXF files with closed paths, proper scaling (millimeters, 1:1), and validated geometry
- **Quality Validation**: Ensures all paths are closed, minimum feature sizes are met, and geometry is suitable for laser cutting
- **Configurable Parameters**: All processing parameters adjustable via YAML config or CLI arguments
- **Preview System**: Multi-stage visualization showing preprocessing, edge detection, contours, and final paths

## Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

```bash
# Basic usage
python src/main.py input_image.jpg output.dxf

# With preview generation
python src/main.py input.jpg output.dxf --preview

# Interactive mode (adjust parameters)
python src/main.py input.jpg output.dxf --interactive

# Override image type detection
python src/main.py input.jpg output.dxf --type photo

# Adjust edge detection thresholds
python src/main.py input.jpg output.dxf --canny-low 30 --canny-high 100

# Specify DPI for scaling
python src/main.py input.jpg output.dxf --dpi 300
```

## Configuration

All processing parameters can be configured in `config/defaults.yaml`:

- **Image Classification**: Automatic photo vs drawing detection
- **Preprocessing**: Noise removal, contrast enhancement, thresholding
- **Edge Detection**: Canny edge detection parameters
- **Vectorization**: Path simplification, circle/arc detection
- **Validation**: Minimum feature sizes, maximum entity counts
- **DXF Output**: Units, layers, coordinate precision

## Processing Pipeline

```
Image Input
    ↓
Image Classifier → Detect photo vs drawing
    ↓
Adaptive Preprocessor → Clean and binarize image
    ↓
Edge Detector → Canny edge detection
    ↓
Contour Extractor → Extract vector paths with hierarchy
    ↓
Path Simplifier → Douglas-Peucker simplification
    ↓
Quality Checker → Validate paths and features
    ↓
DXF Generator → Create laser-ready DXF file
```

## DXF Output Specifications

- **Format**: DXF R2010 (configurable)
- **Units**: Millimeters at 1:1 scale
- **Entities**: Closed LWPolylines (no splines)
- **Layers**: "CutLines" layer (red, color 1)
- **Coordinate System**: All geometry at Z=0 (2D planar)
- **Validation**: All paths closed, minimum 1mm feature size

## Project Structure

```
img_to_dfx/
├── config/
│   └── defaults.yaml          # Configuration parameters
├── src/
│   ├── main.py                # CLI entry point
│   ├── detector/              # Image type classification
│   ├── preprocessing/         # Image preprocessing modules
│   ├── vectorization/         # Edge detection and contour extraction
│   ├── validation/            # Quality checking
│   ├── dxf/                   # DXF file generation
│   ├── preview/               # Visualization
│   └── utils/                 # Utilities (config, logging, scaling)
├── tests/                     # Unit and integration tests
├── examples/                  # Sample images
└── output/                    # Generated DXF files and previews
```

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/ tests/
flake8 src/ tests/
```

## Troubleshooting

### Many small contours rejected
- Decrease `contour_extraction.min_contour_area_pixels` in config
- Or use `--min-area 50` CLI argument

### Edges too fragmented
- Increase `edge_detection.gaussian_blur_sigma`
- Adjust Canny thresholds: `--canny-low 30 --canny-high 100`

### Paths not closing properly
- Check `validation.min_feature_size_mm` - very small features may not close
- Try adjusting `vectorization.simplify_epsilon_mm` (lower = more points)

### DXF file won't open in CAD software
- Ensure all paths are closed (check validation warnings)
- Verify units are set correctly (should be millimeters)
- Try changing DXF version in config (R2000, R2010, R2018)

## License

This project is provided as-is for laser cutting applications.

## Contributing

Contributions welcome! Please test with various image types and ensure DXF output validates correctly.
