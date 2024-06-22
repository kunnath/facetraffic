# Traffic System Detection and Face Recognition

This project processes videos to detect objects using YOLO (You Only Look Once) and recognize faces using OpenCV's Haar Cascades. It downloads a video from YouTube, processes each frame to detect objects and faces, and saves the detected faces as images.

## Features
- Download a video from YouTube.
- Detect objects in the video frames using YOLO.
- Detect faces in the video frames using Haar Cascades.
- Save detected faces as individual image files.

## Requirements
- Python 3.9+
- OpenCV
- numpy
- pytube

## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/trafficsystem.git
    cd trafficsystem
    ```

2. **Install the required packages:**

    ```sh
    pip install opencv-python numpy pytube
    ```

3. **Download necessary files:**
   - YOLO weights file: `yolov3.weights`
   - YOLO config file: `yolov3.cfg`
   - YOLO classes file: `coco.names`

   Ensure these files are in the same directory as your script.

4. **Download the Haar Cascade file:**

   The script will automatically download the `haarcascade_frontalface_default.xml` file from the OpenCV GitHub repository.

## Usage

1. **Modify the script:**
   
   Update the `video_url` variable in the script with a valid YouTube video URL.

   ```python
   video_url = 'https://www.youtube.com/watch?v=abcdefghij'  # Replace with a valid YouTube video URL