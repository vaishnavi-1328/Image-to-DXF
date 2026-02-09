#!/bin/bash
# Launch the Streamlit web application

echo "Starting Image to DXF Converter..."
echo "=================================="
echo ""
echo "The application will open in your browser."
echo "If it doesn't open automatically, go to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

cd "$(dirname "$0")"
streamlit run app.py
