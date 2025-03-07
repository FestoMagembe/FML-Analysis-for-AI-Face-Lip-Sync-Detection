# real_time_lip_sync.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import pickle
from face_mesh_detector import FaceMeshDetector

class LipSyncDetector:
    def __init__(self, model_path, scaler_path):
        # Load the model
        self.model = load_model(model_path)
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Initialize face mesh detector
        self.detector = FaceMeshDetector(static_image_mode=False)
        
        # For smoothing predictions
        self.prediction_history = []
        self.history_size = 5
    
    def process_frame(self, frame):
        """Process a single frame and detect lip sync."""
        # Extract face landmarks
        landmarks = self.detector.detect_landmarks(frame)
        
        # Extract lip landmarks
        lip_landmarks = self.detector.extract_lips_landmarks(landmarks)
        
        # Visualize landmarks
        vis_frame = frame.copy()
        if len(lip_landmarks) > 0:
            vis_frame = self.detector.visualize_landmarks(vis_frame, lip_landmarks)
            
            # Prepare features for prediction
            features = lip_landmarks.flatten()
            
            # Normalize features
            features_scaled = self.scaler.transform([features])[0]
            
            # Make prediction
            prediction = self.model.predict(np.array([features_scaled]))[0][0]
            
            # Smooth predictions
            self.prediction_history.append(prediction)
            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)
            
            smoothed_prediction = np.mean(self.prediction_history)
            
            # Determine if speaking
            is_speaking = smoothed_prediction > 0.5
            
            # Add prediction to visualization
            label = "Speaking" if is_speaking else "Not Speaking"
            color = (0, 255, 0) if is_speaking else (0, 0, 255)
            cv2.putText(vis_frame, f"{label}: {smoothed_prediction:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(vis_frame, "No face detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return vis_frame
    
    def run_webcam(self):
        """Run lip sync detection on webcam feed."""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result_frame = self.process_frame(frame)
            
            # Display result
            cv2.imshow("Lip Sync Detection", result_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_video(self, video_path, output_path=None):
        """Process a video file."""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result_frame = self.process_frame(frame)
            
            # Write frame if output path is provided
            if writer:
                writer.write(result_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        if writer:
            writer.release()

# Example usage
if __name__ == "__main__":
    detector = LipSyncDetector("models/lip_sync_model_best.h5", "processed_data/scaler.pkl")
    
    # Run on webcam
    detector.run_webcam()
    
    # Or process a video file
    # detector.process_video("path/to/video.mp4", "output_video.mp4")
