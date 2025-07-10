from picamera2 import Picamera2
import cv2
import torch
import numpy as np
from flask import Flask, jsonify, render_template, Response, request, redirect, url_for, flash, session
import threading
import telepot  # For Telegram bot
import qrcode
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flash messages

# Global variables to store people count, risk level, heatmap data, and thresholds
people_count = 0
risk_level = "Low"
heatmap_data = []  # Store center points of detected people
threshold_low = 20  # Default low threshold
threshold_medium = 50  # Default medium threshold

# Initialize the Raspberry Pi Camera with Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize Telegram bot
bot_token = ''  # Replace with your Telegram bot token
chat_id = ''  # Replace with your Telegram chat ID
bot = telepot.Bot(bot_token)


def detect_people():
    global people_count, risk_level, heatmap_data, threshold_low, threshold_medium
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
        
        # Use customizable thresholds to determine risk level
        if person_count < threshold_low:
            risk_level = "Low"
        elif threshold_low <= person_count <= threshold_medium:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Send Telegram alert if risk level changes to "High"
        if risk_level == "High" and last_risk_level != "High":
            # Save the frame as an image file
            image_path = "alert_image.jpg"
            
            # Convert the frame from XRGB8888 to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Save the frame as an image file
            cv2.imwrite(image_path, frame_bgr)
            
            # Send the alert message and image via Telegram bot
            with open(image_path, 'rb') as photo:
                bot.sendPhoto(chat_id, photo, caption=f"⚠️ High Crowd Density Alert! {people_count} people detected.")
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


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    global threshold_low, threshold_medium
    if request.method == 'POST':
        try:
            # Update thresholds from form data
            new_threshold_low = int(request.form['threshold_low'])
            new_threshold_medium = int(request.form['threshold_medium'])
            # Validate thresholds
            if new_threshold_low < 0 or new_threshold_medium < 0:
                raise ValueError("Thresholds must be non-negative.")
            if new_threshold_low >= new_threshold_medium:
                raise ValueError("Low threshold must be less than medium threshold.")
            # Update global variables
            threshold_low = new_threshold_low
            threshold_medium = new_threshold_medium
            # Flash a success message
            flash('Thresholds updated successfully!', 'success')
        except ValueError as e:
            # Flash an error message
            flash(f'Error: {e}', 'error')
        # Redirect to the dashboard
        return redirect(url_for('index'))
    # Render the settings page
    return render_template('settings.html', threshold_low=threshold_low, threshold_medium=threshold_medium)


@app.route('/crowd-info')
def crowd_info():
    return render_template('crowd_info.html', people_count=people_count, risk_level=risk_level)


def generate_qr_code():
    # Define the URL for the QR code
    qr_url = f"http://{get_ip_address()}:5000/crowd-info"
    
    # Generate the QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_url)
    qr.make(fit=True)
    
    # Create an image from the QR code
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save the QR code as a file
    img.save("qr_code.png")
    print(f"QR Code generated: {qr_url}")


def get_ip_address():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to a public IP address to determine the local IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


if __name__ == '__main__':
    generate_qr_code()
    # Start the people detection thread
    detection_thread = threading.Thread(target=detect_people, daemon=True)
    detection_thread.start()
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
