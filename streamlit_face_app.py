import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
import subprocess
import sys
from PIL import Image
import glob
from datetime import datetime
import shutil

# Configure Streamlit page
st.set_page_config(
    page_title="Face Detection Video Analyzer",
    page_icon="👥",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin: 1rem 0;
    }
    .face-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        text-align: center;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = []
if 'current_faces' not in st.session_state:
    st.session_state.current_faces = []

def install_and_import_yt_dlp():
    """Install and import yt-dlp if not available"""
    try:
        import yt_dlp
        return yt_dlp
    except ImportError:
        with st.spinner("Installing yt-dlp for video downloading..."):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
        import yt_dlp
        return yt_dlp

def download_haar_cascade():
    """Download Haar cascade file if not exists"""
    haar_cascade_path = os.path.join("./trafficsystem", "haarcascade_frontalface_default.xml")
    
    if not os.path.exists(haar_cascade_path):
        os.makedirs("./trafficsystem", exist_ok=True)
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        
        with st.spinner("Downloading face detection model..."):
            urllib.request.urlretrieve(url, haar_cascade_path)
        
        st.success("Face detection model downloaded successfully!")
    
    return haar_cascade_path

def download_video_with_ytdlp(url, output_path):
    """Download video using yt-dlp"""
    try:
        yt_dlp = install_and_import_yt_dlp()
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'video')
            duration = info.get('duration', 0)
            
            st.info(f"📹 Video: {title}")
            st.info(f"⏱️ Duration: {duration // 60}:{duration % 60:02d}")
            
            # Download the video
            with st.spinner(f"Downloading '{title}'..."):
                ydl.download([url])
            
            # Find the downloaded file
            mp4_files = glob.glob(os.path.join(output_path, "*.mp4"))
            if mp4_files:
                latest_file = max(mp4_files, key=os.path.getctime)
                return latest_file, title
            
        return None, None
        
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return None, None

def detect_and_save_faces(frame, face_cascade, output_folder, frame_count):
    """Detect faces in frame and save them"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    face_count = 0
    for i, (x, y, w, h) in enumerate(faces):
        # Add some padding around the face
        padding = 20
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(frame.shape[1], x + w + padding)
        y_end = min(frame.shape[0], y + h + padding)
        
        face = frame[y_start:y_end, x_start:x_end]
        face_filename = os.path.join(output_folder, f"frame{frame_count:04d}_face{i:02d}.png")
        cv2.imwrite(face_filename, face)
        face_count += 1
    
    return face_count

def process_video_for_faces(video_path, face_cascade_path, output_folder, progress_bar, status_text):
    """Process video to extract faces"""
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error: Failed to open video file.")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    total_faces = 0
    
    # Process every 30th frame to speed up processing
    frame_skip = 30
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            faces_in_frame = detect_and_save_faces(frame, face_cascade, output_folder, frame_count)
            total_faces += faces_in_frame
            
            # Update progress
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames} - Found {total_faces} faces so far...")
        
        frame_count += 1
    
    cap.release()
    return total_faces

def load_face_images(face_folder):
    """Load all face images from the folder"""
    if not os.path.exists(face_folder):
        return []
    
    face_files = [f for f in os.listdir(face_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    face_files.sort()  # Sort by filename
    
    faces = []
    for face_file in face_files:
        face_path = os.path.join(face_folder, face_file)
        try:
            # Load image using PIL
            img = Image.open(face_path)
            faces.append({
                'filename': face_file,
                'image': img,
                'path': face_path
            })
        except Exception as e:
            st.warning(f"Could not load {face_file}: {str(e)}")
    
    return faces

def main():
    # Header
    st.markdown('<h1 class="main-header">👥 Face Detection Video Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Extract and view faces from YouTube videos using AI-powered face detection")
    
    # Sidebar for controls
    st.sidebar.header("🎮 Controls")
    
    # URL Input
    video_url = st.sidebar.text_input(
        "🔗 YouTube Video URL:",
        placeholder="https://youtu.be/example...",
        help="Enter a YouTube video URL to analyze"
    )
    
    # Processing options
    st.sidebar.subheader("⚙️ Processing Options")
    clear_previous = st.sidebar.checkbox("Clear previous results", value=True)
    
    # Process button
    process_button = st.sidebar.button("🚀 Process Video", type="primary")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<h2 class="sub-header">📊 Statistics</h2>', unsafe_allow_html=True)
        
        # Stats placeholder
        stats_placeholder = st.empty()
        
        # Recent videos
        st.markdown('<h3 class="sub-header">📝 Recent Videos</h3>', unsafe_allow_html=True)
        recent_placeholder = st.empty()
    
    with col2:
        st.markdown('<h2 class="sub-header">👤 Detected Faces</h2>', unsafe_allow_html=True)
        faces_placeholder = st.empty()
    
    # Process video when button is clicked
    if process_button and video_url:
        if clear_previous:
            # Clear previous faces
            face_folder = './downloaded_videos/faces'
            if os.path.exists(face_folder):
                shutil.rmtree(face_folder)
            os.makedirs(face_folder, exist_ok=True)
        
        # Setup directories
        output_path = './downloaded_videos'
        face_output_folder = os.path.join(output_path, "faces")
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(face_output_folder, exist_ok=True)
        
        # Download Haar cascade
        face_cascade_path = download_haar_cascade()
        
        # Download video
        video_path, title = download_video_with_ytdlp(video_url, output_path)
        
        if video_path and os.path.exists(video_path):
            st.success(f"✅ Video downloaded successfully!")
            
            # Process video for faces
            st.info("🔍 Processing video for face detection...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_faces = process_video_for_faces(
                video_path, face_cascade_path, face_output_folder, 
                progress_bar, status_text
            )
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            # Add to session state
            st.session_state.processed_videos.append({
                'title': title,
                'url': video_url,
                'faces_count': total_faces,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.success(f"🎉 Found and saved {total_faces} faces!")
            
        else:
            st.error("❌ Failed to download video. Please check the URL.")
    
    # Load and display current faces
    face_folder = './downloaded_videos/faces'
    faces = load_face_images(face_folder)
    st.session_state.current_faces = faces
    
    # Update statistics
    with stats_placeholder.container():
        if faces:
            st.markdown(f"""
            <div class="stats-box">
                <h4>📈 Current Session Stats</h4>
                <p><strong>Total Faces:</strong> {len(faces)}</p>
                <p><strong>Storage Location:</strong> {face_folder}</p>
                <p><strong>Last Updated:</strong> {datetime.now().strftime("%H:%M:%S")}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No faces detected yet. Process a video to see statistics.")
    
    # Update recent videos
    with recent_placeholder.container():
        if st.session_state.processed_videos:
            for i, video in enumerate(reversed(st.session_state.processed_videos[-5:])):
                st.text(f"📹 {video['title'][:30]}...")
                st.text(f"   👥 {video['faces_count']} faces | ⏰ {video['timestamp']}")
        else:
            st.info("No videos processed yet.")
    
    # Display faces
    with faces_placeholder.container():
        if faces:
            # Face viewing options
            cols_per_row = st.slider("Faces per row:", 2, 8, 4)
            face_size = st.slider("Face size:", 50, 200, 100)
            
            # Filter options
            show_frame_info = st.checkbox("Show frame information", True)
            
            # Display faces in grid
            cols = st.columns(cols_per_row)
            
            for i, face_data in enumerate(faces):
                col_idx = i % cols_per_row
                
                with cols[col_idx]:
                    # Display the face image
                    st.image(
                        face_data['image'], 
                        width=face_size,
                        caption=face_data['filename'] if show_frame_info else None
                    )
                    
                    # Add download button for individual face
                    if st.button(f"💾", key=f"download_{i}", help="Download this face"):
                        # Convert PIL image to bytes for download
                        import io
                        img_bytes = io.BytesIO()
                        face_data['image'].save(img_bytes, format='PNG')
                        img_bytes.seek(0)
                        
                        st.download_button(
                            label="Download Face",
                            data=img_bytes.getvalue(),
                            file_name=face_data['filename'],
                            mime="image/png",
                            key=f"download_btn_{i}"
                        )
        else:
            st.info("👆 Enter a YouTube URL and click 'Process Video' to detect faces!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🤖 Powered by OpenCV Face Detection | Built with Streamlit</p>
        <p>💡 Tip: Use videos with clear face visibility for best results</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()