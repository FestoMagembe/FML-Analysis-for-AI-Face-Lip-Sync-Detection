import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Indices for lips landmarks in MediaPipe Face Mesh
        self.lips_indices = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
        ]
    
    def detect_landmarks(self, image):
        """Detect facial landmarks in an image."""
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and get facial landmarks
        results = self.face_mesh.process(image_rgb)
        
        landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                    
        return np.array(landmarks)
    
    def extract_lips_landmarks(self, landmarks):
        """Extract only the lip landmarks."""
        if len(landmarks) == 0:
            return np.array([])
        
        lips_landmarks = np.array([landmarks[i] for i in self.lips_indices])
        return lips_landmarks
    
    def visualize_landmarks(self, image, landmarks):
        """Visualize landmarks on the image."""
        h, w, _ = image.shape
        for landmark in landmarks:
            x, y = int(landmark[0] * w), int(landmark[1] * h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        return image
