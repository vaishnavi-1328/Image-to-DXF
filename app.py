"""
Streamlit Web Application for Image to DXF Conversion
Upload an image, specify metal sheet dimensions, and download a scaled DXF file.
"""
import streamlit as st
import sys
from pathlib import Path
import tempfile
import numpy as np
import cv2
from io import BytesIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config_loader import load_config
from utils.scale_calculator import ScaleCalculator
from preprocessing.drawing_preprocessor import DrawingPreprocessor
from preprocessing.photo_preprocessor import PhotoPreprocessor
from preprocessing.perspective_corrector import PerspectiveCorrector
from vectorization.edge_detector import EdgeDetector
from vectorization.contour_extractor import ContourExtractor
from vectorization.path_simplifier import PathSimplifier
from validation.quality_checker import QualityChecker
from dxf.generator import DXFGenerator

# Import streamlit-image-coordinates for point selection
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_IMAGE_COORDINATES = True
except ImportError:
    HAS_IMAGE_COORDINATES = False


# Page configuration
st.set_page_config(
    page_title="Image to DXF Converter",
    page_icon="🔧",
    layout="wide"
)

# Title
st.title("🔧 Image to DXF Converter for Laser Cutting")
st.markdown("Convert images to DXF files scaled to your metal sheet dimensions")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# Load configuration
@st.cache_resource
def get_config():
    return load_config()

config = get_config()


def process_image(image_data, image_type, sheet_width_mm, sheet_height_mm, fit_mode,
                  corrected_image=None, calculated_dpi=None):
    """
    Process image through the complete pipeline and generate scaled DXF.

    Args:
        image_data: Image bytes
        image_type: 'photo' or 'drawing'
        sheet_width_mm: Target width in millimeters
        sheet_height_mm: Target height in millimeters
        fit_mode: 'fit' or 'fill'
        corrected_image: Optional pre-corrected image (from perspective correction)
        calculated_dpi: Optional DPI calculated from perspective correction

    Returns:
        Tuple of (dxf_bytes, preview_images, stats)
    """
    stats = {}
    preview_images = {}

    # Decode image (unless already corrected)
    if corrected_image is not None:
        image = corrected_image
    else:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image")

    stats['input_shape'] = image.shape
    preview_images['original'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 1: Preprocessing
    if image_type == 'photo':
        preprocessor = PhotoPreprocessor(config['photo_preprocessing'])
    else:
        preprocessor = DrawingPreprocessor(config['drawing_preprocessing'])

    binary = preprocessor.preprocess(image)
    preview_images['binary'] = binary

    # Step 2: Edge Detection
    detector = EdgeDetector(config['edge_detection'])
    edges = detector.detect_edges(binary)
    preview_images['edges'] = edges
    stats['edge_pixels'] = int(np.sum(edges > 0))

    # Step 3: Contour Extraction
    extractor = ContourExtractor(config['contour_extraction'])
    contours_px, _ = extractor.extract_contours(edges)
    stats['num_contours'] = len(contours_px)
    stats['total_points_original'] = sum(len(cnt) for cnt in contours_px)

    # Visualize contours
    contour_vis = extractor.visualize_contours(image, contours_px)
    preview_images['contours'] = cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB)

    # Step 4: Path Simplification
    # Use temporary scale calculator for simplification
    temp_scale = ScaleCalculator(dpi=96)
    simplifier = PathSimplifier(config['vectorization'], temp_scale)
    simplified_px = simplifier.simplify_contours(contours_px)
    stats['total_points_simplified'] = sum(len(cnt) for cnt in simplified_px)

    # Step 5: Calculate scaling to fit metal sheet
    # Find bounding box of all contours in pixels
    all_points = np.vstack(simplified_px)
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)

    content_width_px = max_x - min_x
    content_height_px = max_y - min_y

    stats['content_size_px'] = (content_width_px, content_height_px)

    # Calculate scale to fit within sheet dimensions
    if fit_mode == 'fit':
        # Fit inside (maintain aspect ratio)
        scale_x = sheet_width_mm / content_width_px
        scale_y = sheet_height_mm / content_height_px
        pixels_per_mm = max(1.0 / scale_x, 1.0 / scale_y)  # Use smaller scale to fit
    else:  # fill
        # Fill sheet (may crop, maintain aspect ratio)
        scale_x = sheet_width_mm / content_width_px
        scale_y = sheet_height_mm / content_height_px
        pixels_per_mm = min(1.0 / scale_x, 1.0 / scale_y)  # Use larger scale to fill

    # Create scale calculator with calculated scale
    scale_calc = ScaleCalculator(pixels_per_mm=pixels_per_mm)

    # Translate contours to origin (0,0) and scale
    contours_mm = []
    for contour in simplified_px:
        # Translate to origin
        translated = contour - np.array([min_x, min_y])
        # Scale to millimeters
        scaled = scale_calc.scale_contour(translated)
        contours_mm.append(scaled)

    # Calculate final dimensions
    final_all_points = np.vstack(contours_mm)
    final_min_x, final_min_y = final_all_points.min(axis=0)
    final_max_x, final_max_y = final_all_points.max(axis=0)
    final_width = final_max_x - final_min_x
    final_height = final_max_y - final_min_y

    stats['final_size_mm'] = (final_width, final_height)
    stats['scale_factor'] = pixels_per_mm
    stats['dpi'] = scale_calc.dpi

    # Step 6: Validation
    checker = QualityChecker(config['validation'])
    is_valid, warnings, errors = checker.validate_contours(contours_mm)

    stats['validation'] = {
        'is_valid': is_valid,
        'warnings': len(warnings),
        'errors': len(errors)
    }
    stats['validation_messages'] = {
        'warnings': [w.message for w in warnings[:5]],
        'errors': [e.message for e in errors[:5]]
    }

    # Step 7: Generate DXF
    generator = DXFGenerator(config['dxf_output'])

    metadata = {
        'SOURCE': 'streamlit_app',
        'SHEET_WIDTH_MM': f"{sheet_width_mm:.2f}",
        'SHEET_HEIGHT_MM': f"{sheet_height_mm:.2f}",
        'ACTUAL_WIDTH_MM': f"{final_width:.2f}",
        'ACTUAL_HEIGHT_MM': f"{final_height:.2f}",
        'NUM_CONTOURS': str(len(contours_mm)),
        'FIT_MODE': fit_mode
    }

    # Create DXF in memory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
        dxf_path = generator.create_dxf(contours_mm, tmp.name, metadata=metadata)

        # Read DXF file to bytes
        with open(dxf_path, 'rb') as f:
            dxf_bytes = f.read()

        # Clean up temp file
        Path(dxf_path).unlink()

    return dxf_bytes, preview_images, stats


# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Input")

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload a photo or line drawing to convert to DXF"
    )

    # Image type
    image_type = st.radio(
        "Image Type",
        options=['drawing', 'photo'],
        help="Drawing: Clean line drawings, CAD exports\nPhoto: Photos with varying lighting"
    )

    st.divider()

    # Perspective Correction Section
    st.subheader("🔄 Perspective Correction (Optional)")

    enable_perspective = st.checkbox(
        "Enable Perspective Correction",
        value=False,
        help="Correct perspective distortion in photos taken at an angle"
    )

    # Initialize session state for perspective correction
    if 'perspective_points' not in st.session_state:
        st.session_state.perspective_points = []
    if 'perspective_corrected_image' not in st.session_state:
        st.session_state.perspective_corrected_image = None
    if 'perspective_dpi' not in st.session_state:
        st.session_state.perspective_dpi = None
    if 'last_click_coords' not in st.session_state:
        st.session_state.last_click_coords = None

    perspective_image = None
    perspective_dpi = None

    if enable_perspective and uploaded_file:
        if not HAS_IMAGE_COORDINATES:
            st.warning("⚠️ streamlit-image-coordinates not installed. Install it with: `pip install streamlit-image-coordinates`")
            st.info("Falling back to manual coordinate entry...")

            # Manual coordinate input fallback
            st.markdown("**Enter 4 corner coordinates (in order: TL, TR, BR, BL):**")
            pt_cols = st.columns(4)
            manual_points = []
            for i, label in enumerate(['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']):
                with pt_cols[i]:
                    x = st.number_input(f"{label} X", min_value=0, key=f"pt{i}_x")
                    y = st.number_input(f"{label} Y", min_value=0, key=f"pt{i}_y")
                    manual_points.append((x, y))

            if st.button("Set Points"):
                st.session_state.perspective_points = manual_points
        else:
            # Interactive point selection with streamlit-image-coordinates
            st.info("""
            **📍 Click 4 corners of your object in this order:**
            1. **Top-Left** corner
            2. **Top-Right** corner
            3. **Bottom-Right** corner
            4. **Bottom-Left** corner

            💡 Tip: Click the actual corners of the tilted object you want to straighten.
            """)

            # Get image bytes for display
            image_bytes = uploaded_file.getvalue()
            nparr = np.frombuffer(image_bytes, np.uint8)
            display_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if display_image is not None:
                # Downscale for display if too large
                display_max = config.get('perspective_correction', {}).get('display_max_dimension', 1200)
                h, w = display_image.shape[:2]
                if max(h, w) > display_max:
                    scale = display_max / max(h, w)
                    display_image = cv2.resize(display_image, None, fx=scale, fy=scale)
                    display_scale = scale
                else:
                    display_scale = 1.0

                # Draw existing points on image
                display_with_points = display_image.copy()

                # Draw lines connecting the points if we have at least 2
                if len(st.session_state.perspective_points) >= 2:
                    scaled_points = []
                    for (x, y) in st.session_state.perspective_points:
                        scaled_points.append((int(x * display_scale), int(y * display_scale)))

                    # Draw lines between consecutive points
                    for i in range(len(scaled_points)):
                        start_point = scaled_points[i]
                        end_point = scaled_points[(i + 1) % len(scaled_points)]
                        cv2.line(display_with_points, start_point, end_point, (0, 255, 255), 2)

                # Draw points with numbers
                for i, (x, y) in enumerate(st.session_state.perspective_points):
                    # Scale points to display size
                    display_x = int(x * display_scale)
                    display_y = int(y * display_scale)
                    cv2.circle(display_with_points, (display_x, display_y), 10, (0, 255, 0), -1)
                    cv2.putText(display_with_points, str(i+1), (display_x-5, display_y-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Convert to RGB for display
                display_rgb = cv2.cvtColor(display_with_points, cv2.COLOR_BGR2RGB)

                # Show interactive image
                clicked = streamlit_image_coordinates(display_rgb, key="perspective_selector")

                # Only process new clicks (not repeats from reruns)
                if clicked is not None and len(st.session_state.perspective_points) < 4:
                    current_coords = (clicked['x'], clicked['y'])

                    # Check if this is a new click (different from last processed click)
                    if current_coords != st.session_state.last_click_coords:
                        # Scale coordinates back to original size
                        orig_x = clicked['x'] / display_scale
                        orig_y = clicked['y'] / display_scale
                        st.session_state.perspective_points.append((orig_x, orig_y))
                        st.session_state.last_click_coords = current_coords
                        st.rerun()

                # Show current points
                st.markdown(f"**Selected: {len(st.session_state.perspective_points)}/4 points**")

                # Show point coordinates for debugging
                if len(st.session_state.perspective_points) > 0:
                    with st.expander("View Selected Points (Debug)"):
                        for i, (x, y) in enumerate(st.session_state.perspective_points):
                            corner_names = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
                            st.text(f"{i+1}. {corner_names[i]}: ({x:.1f}, {y:.1f})")

                col_reset, col_clear = st.columns(2)
                with col_reset:
                    if st.button("Reset Points"):
                        st.session_state.perspective_points = []
                        st.session_state.last_click_coords = None
                        st.session_state.perspective_corrected_image = None
                        st.session_state.perspective_dpi = None
                        st.rerun()

        # Dimension inputs
        if len(st.session_state.perspective_points) == 4:
            st.markdown("**Real-world dimensions of the object:**")
            dim_col1, dim_col2 = st.columns(2)

            with dim_col1:
                real_width = st.number_input(
                    "Width (mm)",
                    min_value=1.0,
                    max_value=10000.0,
                    value=300.0,
                    step=10.0,
                    key="perspective_width"
                )

            with dim_col2:
                real_height = st.number_input(
                    "Height (mm)",
                    min_value=1.0,
                    max_value=10000.0,
                    value=200.0,
                    step=10.0,
                    key="perspective_height"
                )

            # Apply correction button
            if st.button("✓ Apply Perspective Correction", type="secondary"):
                try:
                    # Get original image
                    image_bytes = uploaded_file.getvalue()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    st.info(f"Original image size: {original_image.shape[1]}x{original_image.shape[0]} pixels")

                    # Apply perspective correction
                    corrector = PerspectiveCorrector(
                        original_image,
                        config.get('perspective_correction', {})
                    )
                    corrector.set_source_points(st.session_state.perspective_points)
                    corrector.set_output_dimensions(real_width, real_height)

                    # Check for aspect ratio mismatch
                    mismatch = corrector.check_aspect_ratio_mismatch()
                    if mismatch:
                        st.warning(f"⚠️ Aspect ratio mismatch: {mismatch*100:.1f}% difference between selected region and provided dimensions")

                    # Apply correction
                    corrected = corrector.apply_correction()
                    dpi = corrector.get_calculated_dpi()

                    st.info(f"Corrected image size: {corrected.shape[1]}x{corrected.shape[0]} pixels")

                    # Store in session state
                    st.session_state.perspective_corrected_image = corrected
                    st.session_state.perspective_dpi = dpi

                    st.success(f"✅ Perspective corrected! Calculated DPI: {dpi:.1f}")

                except Exception as e:
                    st.error(f"❌ Correction failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        # Use corrected image if available
        if st.session_state.perspective_corrected_image is not None:
            perspective_image = st.session_state.perspective_corrected_image
            perspective_dpi = st.session_state.perspective_dpi

            # Show before/after comparison
            st.success("✅ Perspective correction applied successfully!")
            st.markdown("### 📸 Before & After Comparison")
            st.markdown("**Left:** Original tilted image | **Right:** Corrected straightened image")
            col_before, col_after = st.columns(2)

            with col_before:
                # Show original tilted image
                orig_bytes = uploaded_file.getvalue()
                orig_nparr = np.frombuffer(orig_bytes, np.uint8)
                orig_img = cv2.imdecode(orig_nparr, cv2.IMREAD_COLOR)

                # Draw the selected quadrilateral on original
                orig_with_box = orig_img.copy()
                points_array = np.array(st.session_state.perspective_points, dtype=np.int32)
                cv2.polylines(orig_with_box, [points_array], True, (0, 255, 0), 3)

                orig_rgb = cv2.cvtColor(orig_with_box, cv2.COLOR_BGR2RGB)
                st.image(orig_rgb, caption="BEFORE: Original (with selected region)", use_container_width=True)

            with col_after:
                # Show corrected image
                preview_rgb = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2RGB)
                st.image(preview_rgb, caption="AFTER: Perspective Corrected", use_container_width=True)

    st.divider()

    # Metal sheet dimensions
    st.subheader("📏 Metal Sheet Dimensions")

    col_w, col_h = st.columns(2)

    with col_w:
        sheet_width = st.number_input(
            "Width (mm)",
            min_value=1.0,
            max_value=10000.0,
            value=200.0,
            step=10.0,
            help="Width of the metal sheet in millimeters"
        )

    with col_h:
        sheet_height = st.number_input(
            "Height (mm)",
            min_value=1.0,
            max_value=10000.0,
            value=150.0,
            step=10.0,
            help="Height of the metal sheet in millimeters"
        )

    # Fit mode
    fit_mode = st.radio(
        "Scaling Mode",
        options=['fit', 'fill'],
        help="Fit: Scale to fit inside sheet (may have margins)\nFill: Scale to fill sheet (may crop edges)"
    )

    # Process button
    # Disable if perspective correction is enabled but not yet applied
    can_process = True
    if enable_perspective and uploaded_file:
        if st.session_state.perspective_corrected_image is None:
            can_process = False
            st.warning("⚠️ Please click 'Apply Perspective Correction' button first!")

    process_button = st.button("🚀 Convert to DXF", type="primary", use_container_width=True, disabled=not can_process)

with col2:
    st.header("📥 Output")

    if uploaded_file and process_button:
        try:
            with st.spinner("Processing image..."):
                # Use corrected image if available from perspective correction
                corrected_img = st.session_state.get('perspective_corrected_image')
                calc_dpi = st.session_state.get('perspective_dpi')

                # Process image
                dxf_bytes, previews, stats = process_image(
                    uploaded_file.getvalue(),
                    image_type,
                    sheet_width,
                    sheet_height,
                    fit_mode,
                    corrected_image=corrected_img,
                    calculated_dpi=calc_dpi
                )

            st.success("✅ Conversion complete!")

            # Show if perspective correction was applied
            if corrected_img is not None:
                st.info(f"🔄 Perspective correction was applied (DPI: {calc_dpi:.1f})")

            # Download button
            st.download_button(
                label="📥 Download DXF File",
                data=dxf_bytes,
                file_name="laser_cut_design.dxf",
                mime="application/dxf",
                use_container_width=True
            )

            # Statistics
            st.divider()
            st.subheader("📊 Statistics")

            stat_col1, stat_col2 = st.columns(2)

            with stat_col1:
                st.metric("Contours", stats['num_contours'])
                st.metric("Edge Pixels", f"{stats['edge_pixels']:,}")
                st.metric("Points (Original)", f"{stats['total_points_original']:,}")

            with stat_col2:
                st.metric("Points (Simplified)", f"{stats['total_points_simplified']:,}")
                final_w, final_h = stats['final_size_mm']
                st.metric("Final Width", f"{final_w:.1f} mm")
                st.metric("Final Height", f"{final_h:.1f} mm")

            # Validation status
            if stats['validation']['errors'] > 0:
                st.warning(f"⚠️ {stats['validation']['errors']} validation errors found")
                with st.expander("View Errors"):
                    for error in stats['validation_messages']['errors']:
                        st.text(f"• {error}")

            if stats['validation']['warnings'] > 0:
                st.info(f"ℹ️ {stats['validation']['warnings']} warnings")
                with st.expander("View Warnings"):
                    for warning in stats['validation_messages']['warnings']:
                        st.text(f"• {warning}")

            # Preview images
            st.divider()
            st.subheader("🖼️ Preview")

            # Add perspective corrected tab if correction was applied
            if corrected_img is not None:
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Original", "Perspective Corrected", "Binary", "Edges", "Contours"
                ])

                with tab1:
                    # Show truly original image (before correction)
                    orig_bytes = uploaded_file.getvalue()
                    orig_nparr = np.frombuffer(orig_bytes, np.uint8)
                    orig_img = cv2.imdecode(orig_nparr, cv2.IMREAD_COLOR)
                    orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                    st.image(orig_rgb, caption="Original Image (Before Correction)", use_container_width=True)

                with tab2:
                    st.image(previews['original'], caption="Perspective Corrected Image", use_container_width=True)

                with tab3:
                    st.image(previews['binary'], caption="Preprocessed Binary", use_container_width=True)

                with tab4:
                    st.image(previews['edges'], caption="Detected Edges", use_container_width=True)

                with tab5:
                    st.image(previews['contours'], caption="Extracted Contours", use_container_width=True)
            else:
                tab1, tab2, tab3, tab4 = st.tabs(["Original", "Binary", "Edges", "Contours"])

                with tab1:
                    st.image(previews['original'], caption="Original Image", use_container_width=True)

                with tab2:
                    st.image(previews['binary'], caption="Preprocessed Binary", use_container_width=True)

                with tab3:
                    st.image(previews['edges'], caption="Detected Edges", use_container_width=True)

                with tab4:
                    st.image(previews['contours'], caption="Extracted Contours", use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

    elif not uploaded_file:
        st.info("👈 Upload an image to get started")
    else:
        st.info("👈 Click 'Convert to DXF' to process")


# Sidebar - Advanced Settings
with st.sidebar:
    st.divider()

    with st.expander("🔧 Advanced Settings"):
        st.markdown("### Edge Detection")

        canny_low = st.slider(
            "Canny Low Threshold",
            min_value=10,
            max_value=200,
            value=config.get('edge_detection.canny_low_threshold', 50),
            help="Lower = more edges detected"
        )

        canny_high = st.slider(
            "Canny High Threshold",
            min_value=50,
            max_value=300,
            value=config.get('edge_detection.canny_high_threshold', 150),
            help="Higher = fewer edges detected"
        )

        st.markdown("### Contour Filtering")

        min_area = st.slider(
            "Minimum Contour Area (pixels)",
            min_value=10,
            max_value=500,
            value=config.get('contour_extraction.min_contour_area_pixels', 100),
            help="Smaller features will be ignored"
        )

        st.markdown("### Path Simplification")

        simplify_epsilon = st.slider(
            "Simplification Epsilon (mm)",
            min_value=0.01,
            max_value=0.5,
            value=config.get('vectorization.simplify_epsilon_mm', 0.05),
            step=0.01,
            help="Higher = fewer points, simpler paths"
        )

        # Update config
        if st.button("Apply Settings"):
            config.set('edge_detection.canny_low_threshold', canny_low)
            config.set('edge_detection.canny_high_threshold', canny_high)
            config.set('contour_extraction.min_contour_area_pixels', min_area)
            config.set('vectorization.simplify_epsilon_mm', simplify_epsilon)
            st.success("Settings applied!")

    st.divider()

    st.markdown("""
    ### 📖 Quick Guide

    1. **Upload** your image (photo or drawing)
    2. **Select** image type (drawing/photo)
    3. **[Optional]** Enable perspective correction:
       - Click 4 corners on the tilted image
       - Enter real dimensions
       - Apply correction
    4. **Enter** metal sheet dimensions
    5. **Choose** scaling mode (fit/fill)
    6. **Click** Convert to DXF
    7. **Download** your DXF file!

    ### ℹ️ About

    This tool converts images to DXF files optimized for laser cutting.

    - **Format**: DXF R2010
    - **Units**: Millimeters (1:1 scale)
    - **Paths**: Closed polylines
    - **Layer**: CutLines (red)
    - **New**: Perspective correction for angled photos
    """)

# Footer
st.divider()
st.caption("Image to DXF Converter v1.0 | Optimized for Laser Cutting")
