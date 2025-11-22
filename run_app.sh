#!/bin/bash

# Face Detection Streamlit App Launcher
echo "🚀 Starting Face Detection Video Analyzer..."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit and dependencies..."
    pip install -r streamlit_requirements.txt
fi

# Create necessary directories
mkdir -p downloaded_videos/faces
mkdir -p trafficsystem

echo "🌟 Launching Streamlit app..."
echo "📱 The app will open in your default browser"
echo "🔗 If it doesn't open automatically, go to: http://localhost:8501"

# Run the streamlit app
streamlit run streamlit_face_app.py