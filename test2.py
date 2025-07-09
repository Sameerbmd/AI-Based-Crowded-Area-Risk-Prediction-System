from picamera2 import Picamera2
import cv2
import torch
import numpy as np
from flask import Flask, jsonify, render_template, Response
import threading
import telepot  # For Telegram bot

app = Flask(__name__)

# Global variables to store people count, risk level, and heatmap data
people_count = 0
risk_level = "Low"
heatmap_data = []  # Store center points of detected people

# Initialize the Raspberry Pi Camera with Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize Telegram bot
bot_token = '8173364146:AAHGbLWEXPpib3z9f8w_bzNd5TWNYE-jdmo'  # Replace with your Telegram bot token
chat_id = '1414195004'  # Replace with your Telegram chat ID
bot = telepot.Bot(bot_token)

def detect_people():
    global people_count, risk_level, heatmap_data
    last_risk_level = "Low"  # Track the previous risk level to avoid duplicate alerts
    while True:
        # Capture a frame
        frame = picam2.capture_array()

        # Perform object detection
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        person_count = 0
        centers = []

        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            if class_id == 0 and confidence > 0.4:  # Class ID 0 corresponds to 'person'
                person_count += 1
                centerX = int((x1 + x2) / 2)
                centerY = int((y1 + y2) / 2)
                centers.append((centerX, centerY))

        # Update global variables
        people_count = person_count
        heatmap_data = centers  # Store the center points for the heatmap
        if person_count < 1:
            risk_level = "Low"
        elif 1 <= person_count <= 3:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Send Telegram alert if risk level changes to "High"
        if risk_level == "High" and last_risk_level != "High":
            bot.sendMessage(chat_id, f"⚠️ High Crowd Density Alert! {people_count} people detected.")
            print(f"Alert sent: High Crowd Density ({people_count} people detected).")

        # Update the last risk level
        last_risk_level = risk_level

def generate_frames():
    """Generate frames for MJPEG streaming."""
    while True:
        # Capture a frame
        frame = picam2.capture_array()

        # Convert the frame from XRGB8888 to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    return jsonify({
        'people_count': people_count,
        'risk_level': risk_level,
        'heatmap_data': heatmap_data  # Include heatmap data
    })

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the people detection thread
    detection_thread = threading.Thread(target=detect_people, daemon=True)
    detection_thread.start()

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
