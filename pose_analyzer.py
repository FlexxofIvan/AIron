import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from visualizer import AdvancedVisualizer

import os
import sys
import cv2
import numpy as np
import mediapipe as mp

# Определяем класс прямо здесь, чтобы не импортировать его из main.py
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            sys.stderr.close()
        finally:
            sys.stderr = self._original_stderr

class PoseAnalyzer:
    def __init__(self):
        with SuppressStderr():
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.visualizer = AdvancedVisualizer()
        
    def calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """Calculate angle between three points"""
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        return angle
    
    def get_landmark_coordinates(self, landmarks, landmark_idx: int, image_shape: Tuple[int, int]) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates"""
        height, width = image_shape[:2]
        landmark = landmarks.landmark[landmark_idx]
        return np.array([landmark.x * width, landmark.y * height])
    
    def analyze_squat(self, landmarks, image_shape: Tuple[int, int]) -> Dict:
        """Analyze squat form"""
        try:
            # Key landmarks for squat analysis
            left_hip = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP, image_shape)
            left_knee = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE, image_shape)
            left_ankle = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE, image_shape)
            right_hip = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP, image_shape)
            right_knee = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE, image_shape)
            right_ankle = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE, image_shape)
            
            left_shoulder = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, image_shape)
            right_shoulder = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, image_shape)
            
            # Calculate knee angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # Calculate hip angles
            left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
            
            # Check squat depth (knee angle < 100 degrees for good depth)
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            proper_depth = avg_knee_angle < 100
            
            # Check knee alignment (knees shouldn't cave inward)
            knee_distance = abs(left_knee[0] - right_knee[0])
            ankle_distance = abs(left_ankle[0] - right_ankle[0])
            knee_alignment = knee_distance >= ankle_distance * 0.8
            
            # Check back straightness
            back_straight = abs(left_hip_angle - right_hip_angle) < 20
            
            # Calculate overall form score
            form_score = 0
            if proper_depth:
                form_score += 40
            if knee_alignment:
                form_score += 35
            if back_straight:
                form_score += 25
            
            return {
                'exercise': 'squat',
                'form_score': form_score,
                'knee_angle': avg_knee_angle,
                'left_knee_angle': left_knee_angle,
                'right_knee_angle': right_knee_angle,
                'left_hip_angle': left_hip_angle,
                'right_hip_angle': right_hip_angle,
                'proper_depth': proper_depth,
                'knee_alignment': knee_alignment,
                'back_straight': back_straight,
                'feedback': self._generate_squat_feedback(proper_depth, knee_alignment, back_straight)
            }
            
        except Exception as e:
            return {'exercise': 'squat', 'error': str(e)}
    
    def analyze_pushup(self, landmarks, image_shape: Tuple[int, int]) -> Dict:
        """Analyze push-up form"""
        try:
            # Key landmarks for push-up analysis
            left_shoulder = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, image_shape)
            left_elbow = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW, image_shape)
            left_wrist = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST, image_shape)
            left_hip = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP, image_shape)
            left_ankle = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE, image_shape)
            
            # Calculate elbow angle
            elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Check body alignment (straight line from shoulders to ankles)
            shoulder_hip_ankle_angle = self.calculate_angle(left_shoulder, left_hip, left_ankle)
            body_straight = 160 < shoulder_hip_ankle_angle < 200
            
            # Check if at bottom of push-up (elbow angle around 90 degrees)
            at_bottom = 70 < elbow_angle < 110
            
            # Check full range of motion
            full_rom = elbow_angle < 120
            
            form_score = 0
            if body_straight:
                form_score += 50
            if at_bottom:
                form_score += 30
            if full_rom:
                form_score += 20
            
            return {
                'exercise': 'pushup',
                'form_score': form_score,
                'elbow_angle': elbow_angle,
                'body_straight': body_straight,
                'at_bottom': at_bottom,
                'full_rom': full_rom,
                'feedback': self._generate_pushup_feedback(body_straight, at_bottom, full_rom)
            }
            
        except Exception as e:
            return {'exercise': 'pushup', 'error': str(e)}
    
    def analyze_downward_dog(self, landmarks, image_shape: Tuple[int, int]) -> Dict:
        """Analyze downward dog yoga pose"""
        try:
            # Key landmarks
            left_wrist = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST, image_shape)
            left_shoulder = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, image_shape)
            left_hip = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP, image_shape)
            left_ankle = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE, image_shape)
            
            # Calculate angles for triangle formation
            arm_angle = self.calculate_angle(left_wrist, left_shoulder, left_hip)
            leg_angle = self.calculate_angle(left_shoulder, left_hip, left_ankle)
            
            # Check for inverted V shape
            proper_triangle = 60 < arm_angle < 100 and 60 < leg_angle < 100
            
            # Check arm and leg straightness
            left_elbow = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW, image_shape)
            left_knee = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE, image_shape)
            
            arm_straight = self.calculate_angle(left_wrist, left_elbow, left_shoulder) > 160
            leg_straight = self.calculate_angle(left_hip, left_knee, left_ankle) > 160
            
            form_score = 0
            if proper_triangle:
                form_score += 50
            if arm_straight:
                form_score += 25
            if leg_straight:
                form_score += 25
            
            return {
                'exercise': 'downward_dog',
                'form_score': form_score,
                'proper_triangle': proper_triangle,
                'arm_straight': arm_straight,
                'leg_straight': leg_straight,
                'feedback': self._generate_downward_dog_feedback(proper_triangle, arm_straight, leg_straight)
            }
            
        except Exception as e:
            return {'exercise': 'downward_dog', 'error': str(e)}
    
    def analyze_warrior_pose(self, landmarks, image_shape: Tuple[int, int]) -> Dict:
        """Analyze warrior pose"""
        try:
            # Key landmarks
            left_hip = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP, image_shape)
            left_knee = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE, image_shape)
            left_ankle = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE, image_shape)
            right_hip = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP, image_shape)
            right_knee = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE, image_shape)
            right_ankle = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE, image_shape)
            
            left_shoulder = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, image_shape)
            right_shoulder = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, image_shape)
            left_wrist = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST, image_shape)
            right_wrist = self.get_landmark_coordinates(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST, image_shape)
            
            # Check lunge depth (front leg should be at ~90 degrees)
            front_leg_angle = min(
                self.calculate_angle(left_hip, left_knee, left_ankle),
                self.calculate_angle(right_hip, right_knee, right_ankle)
            )
            proper_lunge = 80 < front_leg_angle < 100
            
            # Check arm positioning (arms should be extended)
            arms_extended = left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]
            
            # Check torso alignment
            torso_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            torso_straight = torso_angle > 160
            
            form_score = 0
            if proper_lunge:
                form_score += 40
            if arms_extended:
                form_score += 30
            if torso_straight:
                form_score += 30
            
            return {
                'exercise': 'warrior_pose',
                'form_score': form_score,
                'proper_lunge': proper_lunge,
                'arms_extended': arms_extended,
                'torso_straight': torso_straight,
                'feedback': self._generate_warrior_feedback(proper_lunge, arms_extended, torso_straight)
            }
            
        except Exception as e:
            return {'exercise': 'warrior_pose', 'error': str(e)}
    
    def _generate_squat_feedback(self, depth: bool, alignment: bool, back: bool) -> List[str]:
        feedback = []
        if not depth:
            feedback.append("Go deeper - thighs should be parallel to ground")
        if not alignment:
            feedback.append("Keep knees aligned over toes")
        if not back:
            feedback.append("Keep back straight and chest up")
        if depth and alignment and back:
            feedback.append("Excellent squat form!")
        return feedback
    
    def _generate_pushup_feedback(self, body: bool, bottom: bool, rom: bool) -> List[str]:
        feedback = []
        if not body:
            feedback.append("Keep body in straight line from head to heels")
        if not bottom:
            feedback.append("Lower chest closer to ground")
        if not rom:
            feedback.append("Use full range of motion")
        if body and bottom and rom:
            feedback.append("Great push-up form!")
        return feedback
    
    def _generate_downward_dog_feedback(self, triangle: bool, arms: bool, legs: bool) -> List[str]:
        feedback = []
        if not triangle:
            feedback.append("Form an inverted V shape with your body")
        if not arms:
            feedback.append("Straighten your arms")
        if not legs:
            feedback.append("Straighten your legs")
        if triangle and arms and legs:
            feedback.append("Perfect downward dog!")
        return feedback
    
    def _generate_warrior_feedback(self, lunge: bool, arms: bool, torso: bool) -> List[str]:
        feedback = []
        if not lunge:
            feedback.append("Bend front knee to 90 degrees")
        if not arms:
            feedback.append("Raise arms above head")
        if not torso:
            feedback.append("Keep torso upright")
        if lunge and arms and torso:
            feedback.append("Strong warrior pose!")
        return feedback
    
    def process_frame(self, frame: np.ndarray, exercise_mode: str, sensitivity: float) -> Tuple[np.ndarray, Dict]:
        """Process video frame and analyze pose"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        analysis_results = {'detected': False}
        
        if results.pose_landmarks:
            analysis_results['detected'] = True
            analysis_results['landmarks'] = results.pose_landmarks
            
            # Analyze based on exercise mode
            if exercise_mode == "Squats":
                analysis_results.update(self.analyze_squat(results.pose_landmarks, frame.shape))
            elif exercise_mode == "Push-ups":
                analysis_results.update(self.analyze_pushup(results.pose_landmarks, frame.shape))
            elif exercise_mode == "Downward Dog":
                analysis_results.update(self.analyze_downward_dog(results.pose_landmarks, frame.shape))
            elif exercise_mode == "Warrior Pose":
                analysis_results.update(self.analyze_warrior_pose(results.pose_landmarks, frame.shape))
            elif exercise_mode == "Auto-detect":
                # Auto-detection logic would go here
                analysis_results.update({'exercise': 'auto_detect', 'form_score': 0})
            
            # Enhanced visualizations
            frame = self.visualizer.draw_enhanced_skeleton(frame, results.pose_landmarks, analysis_results)
            frame = self.visualizer.draw_angle_measurements(frame, results.pose_landmarks, analysis_results)
            frame = self.visualizer.draw_form_zones(frame, results.pose_landmarks, analysis_results)
        
        return frame, analysis_results
    
    def _add_form_overlay(self, frame: np.ndarray, form_score: int):
        """Add form quality overlay to frame"""
        if form_score >= 80:
            color = (0, 255, 0)  # Green
            status = "EXCELLENT"
        elif form_score >= 60:
            color = (0, 255, 255)  # Yellow
            status = "GOOD"
        else:
            color = (0, 0, 255)  # Red
            status = "NEEDS WORK"
        
        cv2.rectangle(frame, (10, 10), (300, 60), color, -1)
        cv2.putText(frame, f"Form: {status} ({form_score}%)", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)