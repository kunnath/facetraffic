import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import os

# Load YOLO
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()

    # Handle the case where net.getUnconnectedOutLayers() returns scalars
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

def process_video(video_path):
    net, classes, output_layers, colors = load_yolo()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open video file.")
        return

    roi_selected = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not roi_selected:
            roi = cv2.selectROI("Select ROI", frame, fromCenter=False)
            roi_selected = True

        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]

        # Detect objects in the ROI frame
        boxes, confidences, class_ids, indexes = detect_objects(roi_frame, net, output_layers)
        draw_labels(boxes, confidences, class_ids, classes, colors, roi_frame, indexes)

        cv2.imshow('ROI', roi_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main(video_path):
    if video_path:
        process_video(video_path)
    else:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        messagebox.showerror("Error", "No video file path provided.")
        root.mainloop()

if __name__ == "__main__":
    video_file_path = './video.mp4/Lamborghini enter BUSY INDIAN Street  Public REACTIONS-1.mp4'  # Replace with your video file path
    main(video_file_path)
