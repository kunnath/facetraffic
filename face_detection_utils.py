"""
Face Detection Utilities
Simplified version of the notebook functions for Streamlit app
"""

import cv2
import os
import subprocess
import sys
from typing import Optional, Tuple
import glob

def install_yt_dlp():
    """Install yt-dlp if not available"""
    try:
        import yt_dlp
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
            return True
        except Exception as e:
            print(f"Failed to install yt-dlp: {e}")
            return False

def download_video(url: str, output_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Download video using yt-dlp
    Returns: (video_path, title) or (None, None) if failed
    """
    if not install_yt_dlp():
        return None, None
    
    try:
        import yt_dlp
        
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
            
            # Download the video
            ydl.download([url])
            
            # Find the downloaded file
            mp4_files = glob.glob(os.path.join(output_path, "*.mp4"))
            if mp4_files:
                latest_file = max(mp4_files, key=os.path.getctime)
                return latest_file, title
        
        return None, None
        
    except Exception as e:
        print(f"Download error: {e}")
        return None, None

def detect_faces_in_frame(frame, face_cascade, output_folder: str, frame_count: int) -> int:
    """
    Detect faces in a single frame and save them
    Returns: number of faces detected
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    face_count = 0
    for i, (x, y, w, h) in enumerate(faces):
        # Add padding around the face
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

def process_video(video_path: str, face_cascade_path: str, output_folder: str, 
                 progress_callback=None, frame_skip: int = 30) -> int:
    """
    Process video to extract faces
    Returns: total number of faces detected
    """
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Failed to open video file")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    total_faces = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame to speed up processing
            if frame_count % frame_skip == 0:
                faces_in_frame = detect_faces_in_frame(
                    frame, face_cascade, output_folder, frame_count
                )
                total_faces += faces_in_frame
                
                # Update progress if callback provided
                if progress_callback:
                    progress = min(frame_count / total_frames, 1.0)
                    progress_callback(progress, frame_count, total_frames, total_faces)
            
            frame_count += 1
    
    finally:
        cap.release()
    
    return total_faces

def get_video_info(video_path: str) -> dict:
    """Get basic video information"""
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': 0
    }
    
    if info['fps'] > 0:
        info['duration'] = info['total_frames'] / info['fps']
    
    cap.release()
    return info