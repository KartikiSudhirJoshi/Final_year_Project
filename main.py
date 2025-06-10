''' from flask import Flask, render_template, request, Response
import cv2
import os
import numpy as np

app = Flask(__name__)

camera = None

def start_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open webcam.")
            camera = None

def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def generate_frames():
    start_camera()
    while True:
        if camera is None:
            frame = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            break
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    stop_camera()
    return render_template('home.html')

@app.route('/conversion')
def conversion():
    start_camera()
    return render_template('conversion.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    stop_camera()
    return render_template('about.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    stop_camera()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        remark = request.form['remark']
        with open('feedback.txt', 'a') as f:
            f.write(f"Name: {name}, Email: {email}, Remark: {remark}\n")
        return render_template('feedback.html', message="Thank you for your feedback!")
    return render_template('feedback.html')

@app.route('/contact')
def contact():
    stop_camera()
    return render_template('contact.html')

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        stop_camera() '''

from flask import Flask, render_template, request, Response
import cv2
import os
import numpy as np

app = Flask(__name__)

camera = None

def start_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open webcam.")
            camera = None

def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def generate_frames():
    start_camera()
    while True:
        if camera is None:
            frame = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            break
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    stop_camera()
    return render_template('home.html')

@app.route('/conversion')
def conversion():
    start_camera()
    return render_template('conversion.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    stop_camera()
    return render_template('about.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    stop_camera()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        remark = request.form['remark']
        with open('feedback.txt', 'a') as f:
            f.write(f"Name: {name}, Email: {email}, Remark: {remark}\n")
        return render_template('feedback.html', message="Thank you for your feedback!")
    return render_template('feedback.html')

@app.route('/contact')
def contact():
    stop_camera()
    return render_template('contact.html')

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        stop_camera()