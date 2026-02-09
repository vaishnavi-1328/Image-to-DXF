# Streamlit Web Application Guide

## 🎉 Your Image-to-DXF Web App is Ready!

A user-friendly web interface for converting images to scaled DXF files for laser cutting.

## Features

✅ **Image Upload** - Drag & drop or browse for images (PNG, JPG, BMP)
✅ **Image Type Selection** - Optimized processing for photos vs drawings
✅ **Custom Dimensions** - Specify metal sheet width & height
✅ **Smart Scaling** - Automatic scaling to fit or fill your sheet
✅ **Live Preview** - See processing steps (binary, edges, contours)
✅ **Validation** - Quality checks before DXF generation
✅ **Download** - One-click DXF file download
✅ **Advanced Settings** - Fine-tune edge detection and simplification

## Quick Start

### Method 1: Use the Launcher Script

```bash
cd /Users/vaishnavis/Desktop/img_to_dfx
./run_app.sh
```

### Method 2: Direct Command

```bash
cd /Users/vaishnavis/Desktop/img_to_dfx
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## How to Use

### Step 1: Upload Your Image

1. Click **"Browse files"** or drag & drop your image
2. Supported formats: PNG, JPG, JPEG, BMP
3. Can be a photo or line drawing

### Step 2: Configure Settings

**Image Type:**
- **Drawing**: For clean line drawings, CAD exports, sketches
- **Photo**: For photos of gates, elevations, decorative patterns

**Metal Sheet Dimensions:**
- Enter the **Width** and **Height** of your metal sheet in millimeters
- Example: 200mm x 150mm

**Scaling Mode:**
- **Fit**: Scales image to fit inside sheet (maintains aspect ratio, may have margins)
- **Fill**: Scales image to fill sheet (maintains aspect ratio, may crop edges)

### Step 3: Convert

Click **"🚀 Convert to DXF"** button

### Step 4: Download

Click **"📥 Download DXF File"** to save your laser-ready DXF file

## Understanding the Output

### Statistics Panel

- **Contours**: Number of closed paths detected
- **Edge Pixels**: Total edge pixels found
- **Points**: Before and after simplification
- **Final Dimensions**: Actual size in your DXF file

### Preview Tabs

1. **Original**: Your uploaded image
2. **Binary**: Preprocessed black & white version
3. **Edges**: Detected edges
4. **Contours**: Final vector paths (green lines)

### Validation Messages

- **Errors** (🔴): Critical issues that may affect laser cutting
- **Warnings** (🟡): Minor issues, usually safe to proceed
- **Info** (🔵): Informational messages

## Scaling Examples

### Example 1: Business Card

**Input Image**: 1000 x 600 pixels
**Metal Sheet**: 85mm x 55mm
**Mode**: Fit

**Result**: Image scaled to fit within 85x55mm card, maintaining aspect ratio

### Example 2: Decorative Panel

**Input Image**: 2000 x 3000 pixels
**Metal Sheet**: 300mm x 450mm
**Mode**: Fill

**Result**: Image scaled to fill 300x450mm panel, may crop edges slightly

### Example 3: Custom Size

**Input Image**: Any size
**Metal Sheet**: Your custom dimensions
**Mode**: Fit or Fill based on preference

## Advanced Settings

Click **"🔧 Advanced Settings"** in the sidebar to adjust:

### Edge Detection
- **Canny Low Threshold** (10-200): Lower = more edges
- **Canny High Threshold** (50-300): Higher = cleaner edges

### Contour Filtering
- **Minimum Area** (10-500 pixels): Ignore features smaller than this

### Path Simplification
- **Epsilon** (0.01-0.5 mm): Higher = fewer points, simpler paths

## Tips for Best Results

### For Photos:
- Use **Photo** mode
- Higher resolution images work better
- Good lighting and contrast help
- Clear backgrounds recommended

### For Drawings:
- Use **Drawing** mode
- Black lines on white background ideal
- Clean, simple designs work best
- Avoid very thin lines (<2 pixels)

### For Scaling:
- **Fit mode** when you need exact margins
- **Fill mode** when you want to maximize sheet usage
- Add 5-10mm margins to your dimensions for safety
- Test with small sheets first

### Common Issues:

**Too many small contours?**
→ Increase "Minimum Contour Area" in Advanced Settings

**Missing important features?**
→ Decrease "Canny Low Threshold"

**Too many unwanted edges?**
→ Increase "Canny High Threshold"

**DXF file too complex?**
→ Increase "Simplification Epsilon"

## DXF File Specifications

The generated DXF files are ready for laser cutting:

- **Format**: AutoCAD R2010 (DXF AC1024)
- **Units**: Millimeters at 1:1 scale
- **Layer**: "CutLines" (red color, ACI 1)
- **Entities**: Closed LWPolylines
- **Coordinate System**: 2D planar (Z=0)

### Metadata Included:

Each DXF file contains:
- Source: streamlit_app
- Sheet dimensions (width × height)
- Actual content dimensions
- Number of contours
- Fit mode used
- Generation timestamp

## Opening DXF Files

Compatible with:
- ✅ LibreCAD (free)
- ✅ AutoCAD
- ✅ QCAD
- ✅ DraftSight
- ✅ Laser cutting software (RDWorks, LightBurn, etc.)

## Stopping the App

Press **Ctrl+C** in the terminal where the app is running

## Troubleshooting

### App won't start?

```bash
# Install/update Streamlit
pip install --upgrade streamlit

# Run again
streamlit run app.py
```

### Port already in use?

```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Browser doesn't open?

Manually navigate to: `http://localhost:8501`

### Processing errors?

1. Check image file is not corrupted
2. Try a simpler image first
3. Adjust Advanced Settings
4. Check error details in the expander

## Performance Tips

- **Large images**: May take 10-30 seconds to process
- **Complex images**: More contours = longer processing
- **Recommendations**:
  - Keep images under 4000x4000 pixels
  - Use compressed formats (JPG for photos)
  - Simple designs process faster

## Example Workflows

### Workflow 1: Custom Keychain

1. Upload keychain design image
2. Select "Drawing" mode
3. Enter dimensions: 50mm × 30mm
4. Select "Fit" mode
5. Convert & download
6. Send DXF to laser cutter

### Workflow 2: Decorative Gate Panel

1. Upload gate photo
2. Select "Photo" mode
3. Enter dimensions: 1000mm × 2000mm
4. Select "Fill" mode
5. Check preview
6. Adjust settings if needed
7. Convert & download

### Workflow 3: Batch Production

1. Use same settings for multiple images
2. Upload first image → Convert → Download
3. Upload next image → Convert → Download
4. Repeat as needed
5. All DXFs will have same scale settings

## API for Automation (Advanced)

The app uses the same backend as the Python API. For automation:

```python
from app import process_image

# Read image
with open('my_image.jpg', 'rb') as f:
    image_data = f.read()

# Process
dxf_bytes, previews, stats = process_image(
    image_data=image_data,
    image_type='photo',
    sheet_width_mm=200,
    sheet_height_mm=150,
    fit_mode='fit'
)

# Save DXF
with open('output.dxf', 'wb') as f:
    f.write(dxf_bytes)
```

## Support

For issues or questions:
1. Check this guide
2. Review the error messages in the app
3. Try adjusting Advanced Settings
4. Test with a simpler image first

## What's Next?

- ✅ Upload images and convert to DXF
- ✅ Specify exact metal sheet dimensions
- ✅ Download scaled, laser-ready files
- 🎯 Use with your laser cutting machine
- 🎯 Create amazing laser-cut designs!

---

**Your web app is ready to use!** 🎉

Start the app with `./run_app.sh` and begin converting images to DXF files!
