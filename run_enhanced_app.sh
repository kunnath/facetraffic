#!/bin/bash
# Enhanced Streamlit Face & Object Detection App Launcher
# Supports both webcam and YouTube video detection

echo "🎯 Starting AI Object & Face Detection System..."
echo "======================================"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if required files exist
echo "🔍 Checking system requirements..."

# Check for YOLO files (optional)
if [ -f "yolov3.weights" ] && [ -f "yolov3.cfg" ] && [ -f "coco.names" ]; then
    echo "✅ YOLO object detection files found - Full detection mode available"
else
    echo "⚠️  YOLO files not found - Face detection only mode"
    echo "💡 Place yolov3.weights, yolov3.cfg, and coco.names in the project directory for object detection"
fi

# Create necessary directories
echo "📁 Setting up directories..."
mkdir -p downloaded_videos/faces
mkdir -p downloaded_videos/webcam_faces
mkdir -p trafficsystem

echo ""
echo "🚀 Launching Enhanced Streamlit App..."
echo "Features available:"
echo "  📹 Real-time Webcam Detection"
echo "  🎥 YouTube Video Analysis"  
echo "  👥 Face Detection"
echo "  📦 Object Detection (if YOLO files present)"
echo ""
echo "🌐 Access the app at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Start the Streamlit app
streamlit run streamlit_face_app.py --server.port 8501 --server.address localhost