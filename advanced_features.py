import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import mediapipe as mp

class WorkoutAnalytics:
    def __init__(self):
        self.performance_history = []
        self.mistake_patterns = {}
        
    def analyze_workout_patterns(self, session_data: Dict) -> Dict:
        """Analyze workout patterns and provide insights"""
        insights = {
            'improvement_trend': self._calculate_improvement_trend(session_data),
            'consistency_score': self._calculate_consistency_score(session_data),
            'weak_points': self._identify_weak_points(session_data),
            'recommendations': self._generate_recommendations(session_data)
        }
        return insights
    
    def _calculate_improvement_trend(self, session_data: Dict) -> str:
        """Calculate if form is improving over the session"""
        exercises = session_data.get('exercises', {})
        
        total_scores = []
        for exercise_data in exercises.values():
            total_scores.extend(exercise_data.get('form_scores', []))
        
        if len(total_scores) < 5:
            return "insufficient_data"
        
        # Calculate trend using linear regression
        x = np.arange(len(total_scores))
        slope = np.polyfit(x, total_scores, 1)[0]
        
        if slope > 2:
            return "improving"
        elif slope < -2:
            return "declining"
        else:
            return "stable"
    
    def _calculate_consistency_score(self, session_data: Dict) -> float:
        """Calculate form consistency score"""
        exercises = session_data.get('exercises', {})
        
        all_scores = []
        for exercise_data in exercises.values():
            all_scores.extend(exercise_data.get('form_scores', []))
        
        if not all_scores:
            return 0.0
        
        # Consistency is inverse of standard deviation
        std_dev = np.std(all_scores)
        consistency = max(0, 100 - std_dev)
        return consistency
    
    def _identify_weak_points(self, session_data: Dict) -> List[str]:
        """Identify areas that need most improvement"""
        exercises = session_data.get('exercises', {})
        weak_points = []
        
        for exercise_name, exercise_data in exercises.items():
            mistakes = exercise_data.get('common_mistakes', [])
            form_scores = exercise_data.get('form_scores', [])
            
            if form_scores and np.mean(form_scores) < 70:
                weak_points.append(f"{exercise_name}: Low average form score")
            
            if len(mistakes) > 3:
                weak_points.append(f"{exercise_name}: Multiple form issues")
        
        return weak_points
    
    def _generate_recommendations(self, session_data: Dict) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        insights = {
            'trend': self._calculate_improvement_trend(session_data),
            'consistency': self._calculate_consistency_score(session_data),
            'weak_points': self._identify_weak_points(session_data)
        }
        
        if insights['trend'] == 'declining':
            recommendations.append("Take breaks between sets to maintain form quality")
        elif insights['trend'] == 'improving':
            recommendations.append("Great progress! Keep focusing on form over speed")
        
        if insights['consistency'] < 60:
            recommendations.append("Focus on consistent movement patterns")
        
        if len(insights['weak_points']) > 2:
            recommendations.append("Consider practicing one exercise at a time")
        
        return recommendations

class BiomechanicalAnalysis:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        
    def analyze_movement_efficiency(self, landmarks_sequence: List, exercise: str) -> Dict:
        """Analyze movement efficiency and biomechanics"""
        if len(landmarks_sequence) < 10:
            return {'insufficient_data': True}
        
        analysis = {
            'movement_smoothness': self._calculate_smoothness(landmarks_sequence),
            'energy_efficiency': self._calculate_energy_efficiency(landmarks_sequence, exercise),
            'joint_stability': self._analyze_joint_stability(landmarks_sequence),
            'tempo_analysis': self._analyze_tempo(landmarks_sequence, exercise)
        }
        
        return analysis
    
    def _calculate_smoothness(self, landmarks_sequence: List) -> float:
        """Calculate movement smoothness based on landmark trajectory"""
        # Track key joint movements
        joint_trajectories = {}
        key_joints = [11, 12, 13, 14, 23, 24, 25, 26]  # Major joints
        
        for joint_idx in key_joints:
            trajectory = []
            for landmarks in landmarks_sequence:
                if landmarks:
                    landmark = landmarks.landmark[joint_idx]
                    trajectory.append([landmark.x, landmark.y])
            joint_trajectories[joint_idx] = np.array(trajectory)
        
        # Calculate smoothness as inverse of acceleration variance
        smoothness_scores = []
        for trajectory in joint_trajectories.values():
            if len(trajectory) > 2:
                velocity = np.diff(trajectory, axis=0)
                acceleration = np.diff(velocity, axis=0)
                smoothness = 1 / (1 + np.var(acceleration))
                smoothness_scores.append(smoothness)
        
        return np.mean(smoothness_scores) * 100 if smoothness_scores else 0
    
    def _calculate_energy_efficiency(self, landmarks_sequence: List, exercise: str) -> float:
        """Calculate energy efficiency of movement"""
        # Simplified efficiency calculation based on unnecessary movements
        if exercise == 'squat':
            return self._squat_efficiency(landmarks_sequence)
        elif exercise == 'pushup':
            return self._pushup_efficiency(landmarks_sequence)
        return 50.0
    
    def _squat_efficiency(self, landmarks_sequence: List) -> float:
        """Calculate squat movement efficiency"""
        hip_trajectories = []
        for landmarks in landmarks_sequence:
            if landmarks:
                left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
                hip_trajectories.append([left_hip.x, left_hip.y])
        
        hip_trajectories = np.array(hip_trajectories)
        
        # Efficient squat should have minimal horizontal hip movement
        horizontal_variance = np.var(hip_trajectories[:, 0])
        efficiency = max(0, 100 - (horizontal_variance * 1000))
        return efficiency
    
    def _pushup_efficiency(self, landmarks_sequence: List) -> float:
        """Calculate push-up movement efficiency"""
        shoulder_trajectories = []
        for landmarks in landmarks_sequence:
            if landmarks:
                left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                shoulder_trajectories.append([left_shoulder.x, left_shoulder.y])
        
        shoulder_trajectories = np.array(shoulder_trajectories)
        
        # Efficient push-up should have minimal horizontal shoulder movement
        horizontal_variance = np.var(shoulder_trajectories[:, 0])
        efficiency = max(0, 100 - (horizontal_variance * 1000))
        return efficiency
    
    def _analyze_joint_stability(self, landmarks_sequence: List) -> Dict:
        """Analyze joint stability during movement"""
        stability_scores = {}
        key_joints = {
            'knee': [self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE],
            'hip': [self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP],
            'shoulder': [self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        }
        
        for joint_name, joint_indices in key_joints.items():
            joint_positions = []
            for landmarks in landmarks_sequence:
                if landmarks:
                    positions = []
                    for idx in joint_indices:
                        landmark = landmarks.landmark[idx]
                        positions.append([landmark.x, landmark.y])
                    joint_positions.append(np.mean(positions, axis=0))
            
            if joint_positions:
                joint_positions = np.array(joint_positions)
                # Stability is inverse of position variance
                variance = np.var(joint_positions, axis=0)
                stability = max(0, 100 - np.mean(variance) * 1000)
                stability_scores[joint_name] = stability
        
        return stability_scores
    
    def _analyze_tempo(self, landmarks_sequence: List, exercise: str) -> Dict:
        """Analyze movement tempo and rhythm"""
        if exercise in ['squat', 'pushup']:
            return self._analyze_rep_tempo(landmarks_sequence, exercise)
        return {'tempo_score': 50, 'rhythm_consistency': 50}
    
    def _analyze_rep_tempo(self, landmarks_sequence: List, exercise: str) -> Dict:
        """Analyze repetition tempo"""
        # Track key joint for tempo analysis
        if exercise == 'squat':
            joint_idx = self.mp_pose.PoseLandmark.LEFT_HIP
        else:  # pushup
            joint_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER
        
        positions = []
        for landmarks in landmarks_sequence:
            if landmarks:
                landmark = landmarks.landmark[joint_idx]
                positions.append(landmark.y)  # Y-position for vertical movement
        
        if len(positions) < 20:
            return {'tempo_score': 50, 'rhythm_consistency': 50}
        
        # Find peaks and valleys to identify rep phases
        positions = np.array(positions)
        # Smooth the signal
        window_size = min(5, len(positions) // 4)
        if window_size >= 3:
            positions = np.convolve(positions, np.ones(window_size)/window_size, mode='same')
        
        # Simple tempo analysis based on position changes
        velocity = np.diff(positions)
        tempo_score = min(100, max(0, 100 - np.std(velocity) * 1000))
        
        # Rhythm consistency based on velocity pattern regularity
        rhythm_score = min(100, max(0, 100 - np.var(velocity) * 1000))
        
        return {
            'tempo_score': tempo_score,
            'rhythm_consistency': rhythm_score
        }

class MistakeHeatmap:
    def __init__(self):
        self.mistake_zones = {}
        
    def update_mistake_zones(self, landmarks, mistake_type: str, image_shape: Tuple[int, int]):
        """Update heatmap data for mistake zones"""
        if mistake_type not in self.mistake_zones:
            self.mistake_zones[mistake_type] = []
        
        # Store landmark positions when mistakes occur
        positions = []
        for landmark in landmarks.landmark:
            positions.append([landmark.x * image_shape[1], landmark.y * image_shape[0]])
        
        self.mistake_zones[mistake_type].append(positions)
    
    def generate_heatmap(self, mistake_type: str, image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate heatmap visualization for specific mistake type"""
        if mistake_type not in self.mistake_zones:
            return np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        
        # Create heatmap from mistake positions
        heatmap = np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)
        
        for mistake_instance in self.mistake_zones[mistake_type]:
            for position in mistake_instance:
                x, y = int(position[0]), int(position[1])
                if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                    # Add gaussian blur around mistake position
                    cv2.circle(heatmap, (x, y), 30, 1.0, -1)
        
        # Normalize and convert to heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap_colored