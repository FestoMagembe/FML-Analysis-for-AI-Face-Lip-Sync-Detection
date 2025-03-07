import cv2
import os
import numpy as np
from face_mesh_detector import FaceMeshDetector

class DataCollector:
    def __init__(self, output_dir="dataset"):
        self.output_dir = output_dir
        self.detector = FaceMeshDetector()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "speaking"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "not_speaking"), exist_ok=True)
    
    def extract_features_from_video(self, video_path, label, max_frames=100):
        """Extract lip landmarks from video frames."""
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        features = []
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract face landmarks
            landmarks = self.detector.detect_landmarks(frame)
            
            # Extract lip landmarks
            lip_landmarks = self.detector.extract_lips_landmarks(landmarks)
            
            if len(lip_landmarks) > 0:
                # Flatten the landmarks and save as features
                flat_landmarks = lip_landmarks.flatten()
                features.append(flat_landmarks)
                
                # Save visualization for debugging
                vis_frame = self.detector.visualize_landmarks(frame.copy(), lip_landmarks)
                cv2.imwrite(f"{self.output_dir}/{label}/frame_{frame_count}.jpg", vis_frame)
                
                # Save the feature vector
                np.save(f"{self.output_dir}/{label}/features_{frame_count}.npy", flat_landmarks)
            
            frame_count += 1
        
        cap.release()
        return features
    
    def process_dataset(self, dataset_config):
        """Process videos according to the dataset configuration."""
        for video_path, label in dataset_config:
            print(f"Processing {video_path} with label {label}")
            self.extract_features_from_video(video_path, label)

# Example usage
if __name__ == "__main__":
    collector = DataCollector()
    
    # Example dataset configuration
    dataset_config = [
        ("path/to/speaking_video1.mp4", "speaking"),
        ("path/to/speaking_video2.mp4", "speaking"),
        ("path/to/silent_video1.mp4", "not_speaking"),
        ("path/to/silent_video2.mp4", "not_speaking")
    ]
    
    collector.process_dataset(dataset_config)
