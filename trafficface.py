import cv2
import numpy as np
from pytube import YouTube
import os

# Load YOLO
def load_yolo(weights_path, config_path, classes_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except AttributeError:  # for older versions of OpenCV that do not have flatten()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, output_layers, colors

def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

def draw_labels(boxes, confidences, class_ids, classes, colors, img, indexes):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_video(video_path, weights_path, config_path, classes_path):
    net, classes, output_layers, colors = load_yolo(weights_path, config_path, classes_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Failed to open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        boxes, confidences, class_ids, indexes = detect_objects(frame, net, output_layers)
        draw_labels(boxes, confidences, class_ids, classes, colors, frame, indexes)

        cv2.imshow('Traffic Sign Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def download_video(url, output_path):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4').get_highest_resolution()
    stream.download(output_path=output_path)
    print(f"Downloaded: {yt.title}")
    return os.path.join(output_path, stream.default_filename)

def main(video_url, weights_path, config_path, classes_path):
    output_path = './downloaded_videos'
    os.makedirs(output_path, exist_ok=True)
    video_path = download_video(video_url, output_path)
    process_video(video_path, weights_path, config_path, classes_path)

if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=0t_QKAm4-Y8'  # Replace with your YouTube video URL
    yolo_weights = 'yolov3.weights'  # Replace with your YOLO weights file
    yolo_config = 'yolov3.cfg'  # Replace with your YOLO config file
    yolo_classes = 'coco.names'  # Replace with your classes file

    main(video_url, yolo_weights, yolo_config, yolo_classes)