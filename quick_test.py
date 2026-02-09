#!/usr/bin/env python
"""
Quick test - Process an image in one command.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Run the full pipeline test
from test_with_real_image import main

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         QUICK TEST - Image to DXF Pipeline              ║
    ╚══════════════════════════════════════════════════════════╝

    This will:
    1. Create a sample drawing (or use your image)
    2. Process it through the complete pipeline
    3. Save visualizations to output/

    Usage:
      python quick_test.py                    # Use sample image
      python quick_test.py your_image.png     # Use your image
      python quick_test.py your_photo.jpg photo  # Process as photo
    """)

    main()
