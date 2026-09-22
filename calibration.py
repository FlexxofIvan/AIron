import numpy as np
import cv2
from typing import Dict, List, Tuple
import mediapipe as mp

class BodyCalibration:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.body_measurements = {}
        self.calibration_complete = False
        
    def calibrate_body_type(self, landmarks, image_shape: Tuple[int, int]) -> Dict:
        """Calibrate for different body types and proportions"""
        try:
            height, width = image_shape[:2]
            
            # Get key measurements
            measurements = self._calculate_body_measurements(landmarks, image_shape)
            
            # Store calibration data
            self.body_measurements = measurements
            self.calibration_complete = True
            
            # Determine body type classification
            body_type = self._classify_body_type(measurements)
            
            return {
                'calibrated': True,
                'body_type': body_type,
                'measurements': measurements,
                'adjustments': self._get_exercise_adjustments(body_type)
            }
            
        except Exception as e:
            return {'calibrated': False, 'error': str(e)}
    
    def _calculate_body_measurements(self, landmarks, image_shape: Tuple[int, int]) -> Dict:
        """Calculate key body measurements for calibration"""
        height, width = image_shape[:2]
        measurements = {}
        
        # Get landmark coordinates
        def get_coords(landmark_idx):
            landmark = landmarks.landmark[landmark_idx]
            return np.array([landmark.x * width, landmark.y * height])
        
        # Key points
        left_shoulder = get_coords(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_coords(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = get_coords(self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_coords(self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = get_coords(self.mp_pose.PoseLandmark.LEFT_KNEE)
        left_ankle = get_coords(self.mp_pose.PoseLandmark.LEFT_ANKLE)
        
        # Calculate measurements
        measurements['shoulder_width'] = np.linalg.norm(right_shoulder - left_shoulder)
        measurements['hip_width'] = np.linalg.norm(right_hip - left_hip)
        measurements['torso_length'] = np.linalg.norm((left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2)
        measurements['thigh_length'] = np.linalg.norm(left_hip - left_knee)
        measurements['shin_length'] = np.linalg.norm(left_knee - left_ankle)
        measurements['total_leg_length'] = measurements['thigh_length'] + measurements['shin_length']
        
        # Calculate ratios
        measurements['shoulder_hip_ratio'] = measurements['shoulder_width'] / measurements['hip_width']
        measurements['torso_leg_ratio'] = measurements['torso_length'] / measurements['total_leg_length']
        measurements['thigh_shin_ratio'] = measurements['thigh_length'] / measurements['shin_length']
        
        return measurements
    
    def _classify_body_type(self, measurements: Dict) -> str:
        """Classify body type based on measurements"""
        shoulder_hip_ratio = measurements.get('shoulder_hip_ratio', 1.0)
        torso_leg_ratio = measurements.get('torso_leg_ratio', 1.0)
        
        if shoulder_hip_ratio > 1.1:
            return "broad_shoulders"
        elif shoulder_hip_ratio < 0.9:
            return "narrow_shoulders"
        elif torso_leg_ratio > 1.2:
            return "long_torso"
        elif torso_leg_ratio < 0.8:
            return "long_legs"
        else:
            return "proportional"
    
    def _get_exercise_adjustments(self, body_type: str) -> Dict:
        """Get exercise-specific adjustments based on body type"""
        adjustments = {
            "broad_shoulders": {
                "squat": {"knee_alignment_tolerance": 1.2, "depth_adjustment": 0},
                "pushup": {"elbow_angle_tolerance": 15, "body_alignment_tolerance": 10}
            },
            "narrow_shoulders": {
                "squat": {"knee_alignment_tolerance": 0.8, "depth_adjustment": 5},
                "pushup": {"elbow_angle_tolerance": 10, "body_alignment_tolerance": 5}
            },
            "long_torso": {
                "squat": {"hip_angle_tolerance": 15, "depth_adjustment": -5},
                "pushup": {"body_alignment_tolerance": 15}
            },
            "long_legs": {
                "squat": {"depth_adjustment": 10, "knee_angle_tolerance": 15},
                "pushup": {"body_alignment_tolerance": 10}
            },
            "proportional": {
                "squat": {},
                "pushup": {}
            }
        }
        return adjustments.get(body_type, {})
    
    def apply_calibration_adjustments(self, analysis_results: Dict) -> Dict:
        """Apply calibration adjustments to analysis results"""
        if not self.calibration_complete:
            return analysis_results
        
        exercise = analysis_results.get('exercise', '')
        body_type = self._classify_body_type(self.body_measurements)
        adjustments = self._get_exercise_adjustments(body_type)
        exercise_adjustments = adjustments.get(exercise, {})
        
        # Apply adjustments to scoring
        if exercise == 'squat' and exercise_adjustments:
            knee_tolerance = exercise_adjustments.get('knee_alignment_tolerance', 1.0)
            depth_adjustment = exercise_adjustments.get('depth_adjustment', 0)
            
            # Adjust knee angle threshold
            knee_angle = analysis_results.get('knee_angle', 180)
            adjusted_depth_threshold = 100 + depth_adjustment
            analysis_results['proper_depth'] = knee_angle < adjusted_depth_threshold
            
        elif exercise == 'pushup' and exercise_adjustments:
            angle_tolerance = exercise_adjustments.get('elbow_angle_tolerance', 0)
            alignment_tolerance = exercise_adjustments.get('body_alignment_tolerance', 0)
            
            # Adjust tolerances for body type
            elbow_angle = analysis_results.get('elbow_angle', 180)
            analysis_results['at_bottom'] = (70 - angle_tolerance) < elbow_angle < (110 + angle_tolerance)
        
        return analysis_results

class FormComparison:
    def __init__(self):
        self.ideal_poses = self._load_ideal_poses()
    
    def _load_ideal_poses(self) -> Dict:
        """Load ideal pose templates for comparison"""
        # Simplified ideal pose data (in practice, this could be loaded from files)
        return {
            'squat': {
                'knee_angle_range': (80, 100),
                'hip_angle_range': (80, 120),
                'back_straightness': 85
            },
            'pushup': {
                'elbow_angle_range': (80, 100),
                'body_alignment': 180,
                'rom_threshold': 120
            },
            'downward_dog': {
                'arm_angle_range': (70, 90),
                'leg_angle_range': (70, 90),
                'arm_straightness': 170,
                'leg_straightness': 170
            },
            'warrior_pose': {
                'lunge_angle_range': (85, 95),
                'torso_alignment': 170,
                'arm_extension': True
            }
        }
    
    def compare_with_ideal(self, analysis_results: Dict) -> Dict:
        """Compare current pose with ideal form"""
        exercise = analysis_results.get('exercise', '')
        ideal = self.ideal_poses.get(exercise, {})
        
        if not ideal:
            return {'comparison_available': False}
        
        comparison = {'comparison_available': True, 'deviations': []}
        
        if exercise == 'squat':
            knee_angle = analysis_results.get('knee_angle', 180)
            ideal_range = ideal['knee_angle_range']
            
            if knee_angle < ideal_range[0]:
                comparison['deviations'].append(f"Knee angle too acute: {knee_angle:.1f}° (ideal: {ideal_range[0]}-{ideal_range[1]}°)")
            elif knee_angle > ideal_range[1]:
                comparison['deviations'].append(f"Not deep enough: {knee_angle:.1f}° (ideal: {ideal_range[0]}-{ideal_range[1]}°)")
                
        elif exercise == 'pushup':
            elbow_angle = analysis_results.get('elbow_angle', 180)
            ideal_range = ideal['elbow_angle_range']
            
            if not (ideal_range[0] <= elbow_angle <= ideal_range[1]):
                comparison['deviations'].append(f"Elbow angle: {elbow_angle:.1f}° (ideal: {ideal_range[0]}-{ideal_range[1]}°)")
        
        return comparison