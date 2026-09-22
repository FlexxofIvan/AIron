import numpy as np
from typing import Dict, Optional
import mediapipe as mp

class ExerciseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.detection_history = []
        self.history_length = 10
        
    def detect_exercise(self, landmarks, image_shape: tuple) -> Optional[str]:
        """Automatically detect which exercise is being performed"""
        if not landmarks:
            return None
            
        try:
            # Get key landmark positions
            left_wrist = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST, image_shape)
            right_wrist = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST, image_shape)
            left_knee = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE, image_shape)
            right_knee = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE, image_shape)
            left_hip = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP, image_shape)
            right_hip = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP, image_shape)
            left_shoulder = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, image_shape)
            right_shoulder = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, image_shape)
            
            # Calculate average positions
            avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
            avg_knee_y = (left_knee[1] + right_knee[1]) / 2
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2
            avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            
            # Detection logic based on body positioning
            detected_exercise = None
            
            # Downward Dog: hands on ground, hips highest point
            if (avg_wrist_y > avg_shoulder_y and 
                avg_hip_y < avg_shoulder_y and 
                avg_hip_y < avg_knee_y):
                detected_exercise = "Downward Dog"
            
            # Push-up: hands on ground, body horizontal
            elif (avg_wrist_y > avg_shoulder_y and 
                  abs(avg_shoulder_y - avg_hip_y) < 50):
                detected_exercise = "Push-ups"
            
            # Warrior Pose: one knee bent, arms up
            elif (avg_wrist_y < avg_shoulder_y and
                  abs(left_knee[1] - right_knee[1]) > 100):
                detected_exercise = "Warrior Pose"
            
            # Squat: standing with knees bent
            elif (avg_wrist_y < avg_shoulder_y and 
                  avg_knee_y > avg_hip_y and
                  abs(left_knee[1] - right_knee[1]) < 50):
                detected_exercise = "Squats"
            
            # Add to detection history for stability
            self.detection_history.append(detected_exercise)
            if len(self.detection_history) > self.history_length:
                self.detection_history.pop(0)
            
            # Return most common detection in recent history
            if detected_exercise:
                exercise_counts = {}
                for ex in self.detection_history:
                    if ex:
                        exercise_counts[ex] = exercise_counts.get(ex, 0) + 1
                
                if exercise_counts:
                    return max(exercise_counts.items(), key=lambda x: x[1])[0]
            
            return None
            
        except Exception:
            return None
    
    def _get_landmark_coords(self, landmarks, landmark_idx: int, image_shape: tuple) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates"""
        height, width = image_shape[:2]
        landmark = landmarks.landmark[landmark_idx]
        return np.array([landmark.x * width, landmark.y * height])
    
    def reset_history(self):
        """Reset detection history"""
        self.detection_history = []