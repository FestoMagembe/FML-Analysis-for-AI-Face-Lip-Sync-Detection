from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
import base64
import pickle
import io
from PIL import Image
import os
from face_mesh_detector import FaceMeshDetector

app = Flask(__name__)

# Load the model and scaler
model = tf.keras.models.load_model("models/lip_sync_model_best.h5")
with open("processed_data/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Initialize face mesh detector
detector = FaceMeshDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_lip_sync():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read the image
        image_file = request.files['image']
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract face landmarks
        landmarks = detector.detect_landmarks(image)
        
        # Extract lip landmarks
        lip_landmarks = detector.extract_lips_landmarks(landmarks)
        
        result = {
            'speaking': False,
            'confidence': 0.0,
            'landmarks_detected': False
        }
        
        if len(lip_landmarks) > 0:
            # Prepare features for prediction
            features = lip_landmarks.flatten()
            
            # Normalize features
            features_scaled = scaler.transform([features])[0]
            
            # Make prediction
            prediction = float(model.predict(np.array([features_scaled]))[0][0])
            
            # Determine if speaking
            is_speaking = prediction > 0.5
            
            result = {
                'speaking': bool(is_speaking),
                'confidence': prediction,
                'landmarks_detected': True
            }
            
            # Create visualization
            vis_image = detector.visualize_landmarks(image.copy(), lip_landmarks)
            _, buffer = cv2.imencode('.jpg', vis_image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            result['visualization'] = img_str
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For video processing
@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
    
    try:
        # Save the uploaded video
        video_file = request.files['video']
        video_path = "uploads/input_video.mp4"
        os.makedirs("uploads", exist_ok=True)
        video_file.save(video_path)
        
        # Process the video
        output_path = "uploads/output_video.mp4"
        
        # Create a video processor instance
        from real_time_lip_sync import LipSyncDetector
        detector = LipSyncDetector("models/lip_sync_model_best.h5", "processed_data/scaler.pkl")
        
        # Process the video
        detector.process_video(video_path, output_path)
        
        # Return the path to the processed video
        return jsonify({'output_video': output_path})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
