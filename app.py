import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import json
import pyttsx3
import threading
import time
from datetime import datetime
import logging
from collections import deque
from flask import Flask, render_template, request, jsonify, Response, url_for, redirect, flash
import mimetypes
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_and_random_key')

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'aslconverter@gmail.com'  # Replace with sender email
app.config['MAIL_PASSWORD'] = os.environ.get('njpybkbaqvqinhlu')  # Set in environment
app.config['MAIL_DEFAULT_SENDER'] = 'aslconverter@gmail.com'

mail = Mail(app)

from flask import Flask, render_template, request, flash, redirect, url_for
import smtplib
from email.message import EmailMessage

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Replace with your actual Gmail and app password
SENDER_EMAIL = 'aslconverter@gmail.com'
SENDER_PASSWORD = 'njpybkbaqvqinhlu'

# Global objects
asl_recognizer_instance = None
camera_instance = None
prediction_lock = threading.Lock()
shared_current_prediction = ""
shared_prediction_confidence = 0.0  # Ensure global initialization

class ASLRecognizer:
    def __init__(self, model_path, label_encoder_path, config_path):
        try:
            try:
                self.model = tf.keras.models.load_model(model_path)
            except:
                logger.warning(f"Failed to load .keras model, trying .h5 format")
                self.model = tf.keras.models.load_model(model_path.replace('.keras', '.h5'))
            logger.info(f"Model loaded from {model_path} successfully.")

            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info(f"Label encoder loaded from {label_encoder_path} successfully.")

            with open(config_path, 'r') as f:
                self.config = json.load(f)

            self.sequence_length = self.config.get('sequence_length', 10)
            self.class_names = self.config.get('class_names', [])
            logger.info(f"Model config loaded. Sequence length: {self.sequence_length}, Classes: {len(self.class_names)}")

            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )

            self.sequence_buffer = deque(maxlen=self.sequence_length)
            self.prediction_threshold = 0.7
            logger.info("ASL Recognizer initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing ASL Recognizer: {e}")
            raise

    def extract_landmarks(self, image):
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                return np.array(landmarks, dtype=np.float32)
            return np.zeros(63, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Error extracting landmarks: {e}")
            return np.zeros(63, dtype=np.float32)

    def normalize_landmarks(self, landmarks):
        if landmarks.sum() == 0:
            return np.zeros(63, dtype=np.float32)
        reshaped = landmarks.reshape(21, 3)
        wrist = reshaped[0]
        normalized = reshaped - wrist
        return normalized.flatten()

    def predict_from_frame(self, frame):
        try:
            landmarks = self.extract_landmarks(frame)
            normalized_landmarks = self.normalize_landmarks(landmarks)
            self.sequence_buffer.append(normalized_landmarks)
            predicted_class = None
            confidence = 0.0

            if len(self.sequence_buffer) == self.sequence_length:
                sequence_input = np.array(list(self.sequence_buffer), dtype=np.float32)
                sequence_input = np.expand_dims(sequence_input, axis=0)
                prediction_probs = self.model.predict(sequence_input, verbose=0)[0]
                predicted_class_idx = np.argmax(prediction_probs)
                confidence = prediction_probs[predicted_class_idx]
                if confidence > self.prediction_threshold:
                    predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Error in real-time prediction from frame: {e}")
            return None, 0.0

    def predict_from_video(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Could not open video file: {video_path}")
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Processing video '{video_path}' with frame rate: {frame_rate} FPS")
            predictions_list = []
            self.sequence_buffer.clear()
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_skip_interval = max(1, int(frame_rate / 10))
                if frame_count % frame_skip_interval == 0:
                    prediction, confidence = self.predict_from_frame(frame.copy())
                    if prediction:
                        predictions_list.append(prediction)
                frame_count += 1
            cap.release()
            if predictions_list:
                from collections import Counter
                most_common_sign = Counter(predictions_list).most_common(1)[0][0]
                logger.info(f"Video '{video_path}' processed. Most common sign: {most_common_sign}")
                return most_common_sign, 1.0
            logger.info(f"No clear signs detected in video '{video_path}'.")
            return None, 0.0
        except Exception as e:
            logger.error(f"Error processing video '{video_path}': {e}")
            return None, 0.0

def initialize_application_components():
    global asl_recognizer_instance
    logger.info("Initializing ASL Recognition application components...")
    try:
        model_path = "model_outputs/final_asl_model.keras"
        label_encoder_path = "model_outputs/label_encoder.pkl"
        config_path = "model_outputs/model_config.json"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('static/audio', exist_ok=True)
        asl_recognizer_instance = ASLRecognizer(model_path, label_encoder_path, config_path)
        logger.info("ASL Recognizer instance created successfully.")
    except Exception as e:
        logger.error(f"Critical error during application initialization: {e}")
        exit(1)

def text_to_speech(text, filename_prefix):
    try:
        if not text or text.strip() == "":
            logger.error("Empty or invalid text provided for TTS.")
            return None
        
        # Sanitize text for filename
        safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '_')).strip()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        audio_filename = f'{filename_prefix}_{safe_text}_{timestamp}.mp3'
        audio_path_full = os.path.join('static', 'audio', audio_filename)
        audio_dir = os.path.dirname(audio_path_full)

        # Ensure audio directory exists and is writable
        os.makedirs(audio_dir, exist_ok=True)
        if not os.access(audio_dir, os.W_OK):
            logger.error(f"No write permission for directory: {audio_dir}")
            return None

        # Use pyttsx3 only (offline, reliable)
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Adjust speech rate
            engine.setProperty('volume', 0.9)  # Set volume (0.0 to 1.0)
            voices = engine.getProperty('voices')
            for voice in voices:
                if "english" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            # Fallback: play directly if file saving fails
            engine.save_to_file(text, audio_path_full)
            engine.runAndWait()
            logger.info(f"Audio generated with pyttsx3: {audio_path_full}")
        except Exception as pyttsx3_error:
            logger.error(f"pyttsx3 failed: {pyttsx3_error}")
            # Fallback: play audio directly
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                logger.info("Played audio directly via pyttsx3 as fallback.")
            except Exception as fallback_error:
                logger.error(f"Fallback audio playback failed: {fallback_error}")
            return None

        # Verify the audio file exists
        if not os.path.exists(audio_path_full):
            logger.error(f"Audio file was not created at: {audio_path_full}")
            return None

        logger.info(f"Audio saved: {audio_path_full}")
        return url_for('static', filename=f'audio/{audio_filename}', _external=True)
    
    except Exception as e:
        logger.error(f"Unexpected error generating speech for '{text}': {e}")
        return None

def cleanup_old_audio_files(max_age_hours=24):
    audio_dir = os.path.join('static', 'audio')
    now = time.time()
    for filename in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > max_age_hours * 3600:
                os.remove(file_path)
                logger.info(f"Removed old audio file: {file_path}")

def generate_frames():
    global camera_instance, shared_current_prediction, shared_prediction_confidence
    if camera_instance is None:
        camera_instance = cv2.VideoCapture(0)
        if not camera_instance.isOpened():
            logger.error("Failed to open webcam.")
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n' + b'Error: Could not open webcam.' + b'\r\n')
            return
        camera_instance.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_instance.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logger.info("Webcam initialized.")
    if asl_recognizer_instance:
        asl_recognizer_instance.sequence_buffer.clear()
        logger.info("ASL Recognizer sequence buffer cleared.")
    while True:
        success, frame = camera_instance.read()
        if not success:
            logger.warning("Failed to read frame from webcam.")
            break
        frame = cv2.flip(frame, 1)
        if asl_recognizer_instance:
            prediction, confidence = asl_recognizer_instance.predict_from_frame(frame.copy())
            with prediction_lock:
                if prediction:
                    if prediction != shared_current_prediction:
                        shared_current_prediction = prediction
                        shared_prediction_confidence = confidence
                        logger.debug(f"Live Prediction: {prediction} (Confidence: {confidence:.2f})")
                else:
                    shared_current_prediction = ""
                    shared_prediction_confidence = 0.0
        display_prediction = shared_current_prediction if shared_current_prediction else "Detecting..."
        display_confidence = shared_prediction_confidence
        cv2.putText(frame, f"Sign: {display_prediction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if display_confidence > 0:
            cv2.putText(frame, f"Confidence: {display_confidence:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    if camera_instance:
        camera_instance.release()
        logger.info("Webcam released.")
        camera_instance = None

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/conversion', methods=['GET', 'POST'])
def conversion():
    global shared_current_prediction, shared_prediction_confidence  # Declare as global
    text_result = ""
    audio_file_url = None
    error_message = None
    if request.method == 'POST':
        if 'video' in request.files:
            video_file = request.files['video']
            if video_file.filename == '':
                flash('No selected video file.', 'warning')
                return redirect(request.url)
            mime_type, _ = mimetypes.guess_type(video_file.filename)
            if mime_type not in ['video/mp4', 'video/avi']:
                flash('Unsupported video format. Please upload an MP4 or AVI file.', 'error')
                return redirect(request.url)
            if asl_recognizer_instance is None:
                flash("Recognition model not loaded.", 'error')
                return redirect(request.url)
            try:
                upload_dir = 'uploads'
                os.makedirs(upload_dir, exist_ok=True)
                video_filename = f"uploaded_video_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
                video_path = os.path.join(upload_dir, video_filename)
                video_file.save(video_path)
                logger.info(f"Uploaded video saved to: {video_path}")
                prediction, confidence = asl_recognizer_instance.predict_from_video(video_path)
                if prediction and confidence > 0:
                    text_result = prediction.upper()
                    logger.info(f"Calling text_to_speech with text: {text_result}")
                    audio_file_url = text_to_speech(text_result, "video_prediction")
                    if audio_file_url:
                        flash(f"Video processed successfully! Recognized: {text_result}", 'success')
                    else:
                        flash(f"Video processed, but audio generation failed for '{text_result}'. Check logs for details.", 'error')
                else:
                    error_message = "Could not detect any clear sign in the uploaded video."
                    flash(error_message, 'warning')
                if os.path.exists(video_path):
                    os.remove(video_path)
                    logger.info(f"Cleaned up uploaded video: {video_path}")
            except Exception as e:
                logger.error(f"Error processing uploaded video: {e}")
                error_message = f"An error occurred while processing the video: {str(e)}"
                flash(error_message, 'error')
                if os.path.exists(video_path):
                    os.remove(video_path)
                    logger.info(f"Cleaned up uploaded video: {video_path}")
        elif request.form.get('action') == 'predict_webcam_to_text':
            with prediction_lock:
                if shared_current_prediction and shared_prediction_confidence > 0.5:
                    text_result = shared_current_prediction.upper()
                    logger.info(f"Calling text_to_speech for webcam prediction: {text_result}")
                    audio_file_url = text_to_speech(text_result, "webcam_capture")
                    if audio_file_url:
                        flash(f"Webcam capture recognized: {text_result}", 'info')
                    else:
                        flash(f"Webcam prediction processed, but audio generation failed for '{text_result}'. Check logs for details.", 'error')
                else:
                    error_message = "No clear sign detected from webcam."
                    flash(error_message, 'warning')
                shared_current_prediction = ""
                shared_prediction_confidence = 0.0
    return render_template('conversion.html',
                           text=text_result,
                           audio=audio_file_url,
                           error=error_message,
                           shared_prediction_confidence=shared_prediction_confidence)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        remark = request.form.get('remark')
        logger.info(f"Feedback received from {name} ({email}): {remark}")
        flash("Thank you for your feedback!", 'success')
        return redirect(url_for('feedback'))
    return render_template('feedback.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/api/current_prediction')
def current_prediction_api():
    with prediction_lock:
        return jsonify({
            'prediction': shared_current_prediction,
            'confidence': float(shared_prediction_confidence)
        })

if __name__ == '__main__':
    initialize_application_components()
    cleanup_old_audio_files()
    app.run(debug=True, host='0.0.0.0', port=5000)