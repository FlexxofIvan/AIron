import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional
import mediapipe as mp

class AdvancedVisualizer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.colors = {
            'excellent': (0, 255, 0),    # Green
            'good': (0, 255, 255),       # Yellow
            'needs_work': (0, 0, 255),   # Red
            'neutral': (255, 255, 255),  # White
            'joint': (255, 0, 255),      # Magenta
            'angle': (0, 255, 255)       # Cyan
        }
        
    def draw_enhanced_skeleton(self, frame: np.ndarray, landmarks, analysis_results: Dict) -> np.ndarray:
        """Draw enhanced skeleton with color-coded joints based on form quality"""
        if not landmarks:
            return frame
        
        height, width = frame.shape[:2]
        
        # Draw base pose landmarks
        self.mp_drawing.draw_landmarks(
            frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
        )
        
        # Highlight specific joints based on exercise and form quality
        self._highlight_critical_joints(frame, landmarks, analysis_results, (height, width))
        
        return frame
    
    def draw_angle_measurements(self, frame: np.ndarray, landmarks, analysis_results: Dict) -> np.ndarray:
        """Draw real-time angle measurements on frame"""
        if not landmarks or 'exercise' not in analysis_results:
            return frame
        
        height, width = frame.shape[:2]
        exercise = analysis_results['exercise']
        
        if exercise == 'squat':
            self._draw_squat_angles(frame, landmarks, (height, width), analysis_results)
        elif exercise == 'pushup':
            self._draw_pushup_angles(frame, landmarks, (height, width), analysis_results)
        elif exercise in ['downward_dog', 'warrior_pose']:
            self._draw_yoga_angles(frame, landmarks, (height, width), analysis_results)
        
        return frame
    
    def _highlight_critical_joints(self, frame: np.ndarray, landmarks, analysis_results: Dict, image_shape: Tuple[int, int]):
        """Highlight joints that are critical for current exercise"""
        exercise = analysis_results.get('exercise', '')
        form_score = analysis_results.get('form_score', 0)
        
        # Determine color based on form score
        if form_score >= 80:
            joint_color = self.colors['excellent']
        elif form_score >= 60:
            joint_color = self.colors['good']
        else:
            joint_color = self.colors['needs_work']
        
        critical_joints = self._get_critical_joints(exercise)
        
        for joint_idx in critical_joints:
            landmark = landmarks.landmark[joint_idx]
            x = int(landmark.x * image_shape[1])
            y = int(landmark.y * image_shape[0])
            cv2.circle(frame, (x, y), 8, joint_color, -1)
            cv2.circle(frame, (x, y), 10, (0, 0, 0), 2)
    
    def _get_critical_joints(self, exercise: str) -> List[int]:
        """Get list of critical joint indices for each exercise"""
        joint_maps = {
            'squat': [
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE
            ],
            'pushup': [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST,
                self.mp_pose.PoseLandmark.RIGHT_WRIST,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP
            ],
            'downward_dog': [
                self.mp_pose.PoseLandmark.LEFT_WRIST,
                self.mp_pose.PoseLandmark.RIGHT_WRIST,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE
            ],
            'warrior_pose': [
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_WRIST,
                self.mp_pose.PoseLandmark.RIGHT_WRIST
            ]
        }
        return joint_maps.get(exercise, [])
    
    def _draw_squat_angles(self, frame: np.ndarray, landmarks, image_shape: Tuple[int, int], analysis_results: Dict):
        """Draw angle measurements for squat analysis"""
        knee_angle = analysis_results.get('knee_angle', 0)
        
        # Draw knee angle
        left_knee = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE, image_shape)
        self._draw_angle_text(frame, left_knee, f"Knee: {knee_angle:.1f}°", 
                             self.colors['excellent'] if knee_angle < 100 else self.colors['needs_work'])
        
        # Draw depth indicator
        depth_color = self.colors['excellent'] if analysis_results.get('proper_depth') else self.colors['needs_work']
        cv2.putText(frame, f"Depth: {'GOOD' if analysis_results.get('proper_depth') else 'SHALLOW'}", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, depth_color, 2)
    
    def _draw_pushup_angles(self, frame: np.ndarray, landmarks, image_shape: Tuple[int, int], analysis_results: Dict):
        """Draw angle measurements for push-up analysis"""
        elbow_angle = analysis_results.get('elbow_angle', 0)
        
        # Draw elbow angle
        left_elbow = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW, image_shape)
        self._draw_angle_text(frame, left_elbow, f"Elbow: {elbow_angle:.1f}°",
                             self.colors['excellent'] if 70 < elbow_angle < 110 else self.colors['needs_work'])
        
        # Draw body alignment indicator
        alignment_color = self.colors['excellent'] if analysis_results.get('body_straight') else self.colors['needs_work']
        cv2.putText(frame, f"Body: {'STRAIGHT' if analysis_results.get('body_straight') else 'BENT'}", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, alignment_color, 2)
    
    def _draw_yoga_angles(self, frame: np.ndarray, landmarks, image_shape: Tuple[int, int], analysis_results: Dict):
        """Draw angle measurements for yoga poses"""
        exercise = analysis_results.get('exercise', '')
        
        if exercise == 'downward_dog':
            triangle_color = self.colors['excellent'] if analysis_results.get('proper_triangle') else self.colors['needs_work']
            cv2.putText(frame, f"Triangle: {'GOOD' if analysis_results.get('proper_triangle') else 'ADJUST'}", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, triangle_color, 2)
        
        elif exercise == 'warrior_pose':
            lunge_color = self.colors['excellent'] if analysis_results.get('proper_lunge') else self.colors['needs_work']
            cv2.putText(frame, f"Lunge: {'GOOD' if analysis_results.get('proper_lunge') else 'DEEPER'}", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, lunge_color, 2)
    
    def _draw_angle_text(self, frame: np.ndarray, position: np.ndarray, text: str, color: Tuple[int, int, int]):
        """Draw angle text at specific position"""
        x, y = int(position[0]), int(position[1])
        cv2.putText(frame, text, (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (x, y), 5, color, -1)
    
    def _get_landmark_coords(self, landmarks, landmark_idx: int, image_shape: Tuple[int, int]) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates"""
        height, width = image_shape[:2]
        landmark = landmarks.landmark[landmark_idx]
        return np.array([landmark.x * width, landmark.y * height])
    
    def create_form_trend_chart(self, form_scores: List[float], exercise: str) -> plt.Figure:
        """Create form score trend chart"""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        if not form_scores:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{exercise.title()} Form Trend')
            return fig
        
        x = list(range(len(form_scores)))
        ax.plot(x, form_scores, 'b-', linewidth=2, label='Form Score')
        
        # Add threshold lines
        ax.axhline(y=80, color='g', linestyle='--', alpha=0.7, label='Excellent (80%)')
        ax.axhline(y=60, color='y', linestyle='--', alpha=0.7, label='Good (60%)')
        
        # Color-code the area based on performance zones
        ax.fill_between(x, form_scores, 80, where=np.array(form_scores) >= 80, 
                       color='green', alpha=0.3, interpolate=True)
        ax.fill_between(x, form_scores, 60, where=(np.array(form_scores) >= 60) & (np.array(form_scores) < 80), 
                       color='yellow', alpha=0.3, interpolate=True)
        ax.fill_between(x, 0, form_scores, where=np.array(form_scores) < 60, 
                       color='red', alpha=0.3, interpolate=True)
        
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Form Score (%)')
        ax.set_title(f'{exercise.title()} Form Trend')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_rep_progress_visualization(self, current_reps: int, target_reps: int, form_scores: List[float]) -> plt.Figure:
        """Create rep progress visualization with form quality"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
        # Rep progress bar
        progress = min(current_reps / max(target_reps, 1), 1.0) * 100
        ax1.barh(['Reps'], [progress], color='lightblue', height=0.5)
        ax1.barh(['Reps'], [100 - progress], left=[progress], color='lightgray', height=0.5)
        ax1.set_xlim(0, 100)
        ax1.set_title(f'Rep Progress: {current_reps}/{target_reps}')
        ax1.set_xlabel('Progress (%)')
        
        # Form quality per rep
        if form_scores:
            rep_numbers = list(range(1, len(form_scores) + 1))
            colors = ['green' if score >= 80 else 'yellow' if score >= 60 else 'red' for score in form_scores]
            ax2.bar(rep_numbers, form_scores, color=colors, alpha=0.7)
            ax2.set_xlabel('Rep Number')
            ax2.set_ylabel('Form Score (%)')
            ax2.set_title('Form Quality per Rep')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_3d_pose_visualization(self, landmarks, exercise: str) -> plt.Figure:
        """Create 3D visualization of pose"""
        if not landmarks:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.text(0.5, 0.5, 0.5, 'No pose detected', ha='center', va='center')
            return fig
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract 3D coordinates
        points_3d = []
        for landmark in landmarks.landmark:
            points_3d.append([landmark.x, landmark.y, landmark.z])
        points_3d = np.array(points_3d)
        
        # Plot pose landmarks
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c='red', s=50, alpha=0.8)
        
        # Draw connections for major body segments
        connections = [
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),
            # Legs
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]
        
        for start, end in connections:
            if start < len(points_3d) and end < len(points_3d):
                ax.plot([points_3d[start, 0], points_3d[end, 0]],
                       [points_3d[start, 1], points_3d[end, 1]],
                       [points_3d[start, 2], points_3d[end, 2]], 'b-', alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(f'3D Pose - {exercise.title()}')
        
        # Set equal aspect ratio
        max_range = np.array([points_3d[:, 0].max() - points_3d[:, 0].min(),
                             points_3d[:, 1].max() - points_3d[:, 1].min(),
                             points_3d[:, 2].max() - points_3d[:, 2].min()]).max() / 2.0
        mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
        mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
        mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        return fig
    
    def draw_form_zones(self, frame: np.ndarray, landmarks, analysis_results: Dict) -> np.ndarray:
        """Draw colored zones indicating form quality areas"""
        if not landmarks:
            return frame
        
        height, width = frame.shape[:2]
        exercise = analysis_results.get('exercise', '')
        
        # Create overlay
        overlay = frame.copy()
        
        if exercise == 'squat':
            self._draw_squat_zones(overlay, landmarks, (height, width), analysis_results)
        elif exercise == 'pushup':
            self._draw_pushup_zones(overlay, landmarks, (height, width), analysis_results)
        
        # Blend overlay with original frame
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        return frame
    
    def _draw_squat_zones(self, frame: np.ndarray, landmarks, image_shape: Tuple[int, int], analysis_results: Dict):
        """Draw colored zones for squat form"""
        # Knee tracking zone
        left_knee = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE, image_shape)
        right_knee = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE, image_shape)
        left_ankle = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE, image_shape)
        right_ankle = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE, image_shape)
        
        # Draw alignment zones
        knee_alignment_color = self.colors['excellent'] if analysis_results.get('knee_alignment') else self.colors['needs_work']
        
        # Draw rectangles around knee zones
        cv2.rectangle(frame, 
                     (int(left_knee[0]) - 30, int(left_knee[1]) - 30),
                     (int(left_knee[0]) + 30, int(left_knee[1]) + 30),
                     knee_alignment_color, 3)
        cv2.rectangle(frame, 
                     (int(right_knee[0]) - 30, int(right_knee[1]) - 30),
                     (int(right_knee[0]) + 30, int(right_knee[1]) + 30),
                     knee_alignment_color, 3)
    
    def _draw_pushup_zones(self, frame: np.ndarray, landmarks, image_shape: Tuple[int, int], analysis_results: Dict):
        """Draw colored zones for push-up form"""
        # Body alignment zone
        left_shoulder = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, image_shape)
        left_hip = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP, image_shape)
        left_ankle = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE, image_shape)
        
        alignment_color = self.colors['excellent'] if analysis_results.get('body_straight') else self.colors['needs_work']
        
        # Draw body alignment line
        cv2.line(frame, tuple(left_shoulder.astype(int)), tuple(left_ankle.astype(int)), alignment_color, 4)
    
    def create_comparison_visualization(self, current_landmarks, ideal_pose_data: Dict, exercise: str) -> plt.Figure:
        """Create side-by-side comparison of current vs ideal pose"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Current pose
        ax1.set_title('Your Current Pose')
        if current_landmarks:
            self._plot_pose_2d(ax1, current_landmarks, 'blue')
        else:
            ax1.text(0.5, 0.5, 'No pose detected', ha='center', va='center', transform=ax1.transAxes)
        
        # Ideal pose
        ax2.set_title('Ideal Form')
        if ideal_pose_data.get('landmarks'):
            self._plot_pose_2d(ax2, ideal_pose_data['landmarks'], 'green')
        else:
            ax2.text(0.5, 0.5, 'Ideal pose template\ncoming soon', ha='center', va='center', transform=ax2.transAxes)
        
        for ax in [ax1, ax2]:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.invert_yaxis()  # Invert y-axis to match image coordinates
        
        plt.suptitle(f'{exercise.title()} - Form Comparison')
        plt.tight_layout()
        return fig
    
    def _plot_pose_2d(self, ax, landmarks, color: str):
        """Plot 2D pose on matplotlib axes"""
        # Extract x, y coordinates
        x_coords = [landmark.x for landmark in landmarks.landmark]
        y_coords = [landmark.y for landmark in landmarks.landmark]
        
        # Plot landmarks
        ax.scatter(x_coords, y_coords, c=color, s=30, alpha=0.8)
        
        # Draw connections
        connections = [
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # Upper body
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (24, 26), (25, 27), (26, 28)  # Lower body
        ]
        
        for start, end in connections:
            if start < len(x_coords) and end < len(x_coords):
                ax.plot([x_coords[start], x_coords[end]], 
                       [y_coords[start], y_coords[end]], color, alpha=0.7, linewidth=2)
    
    def create_performance_dashboard(self, session_data: Dict) -> plt.Figure:
        """Create comprehensive performance dashboard"""
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall session stats
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_session_overview(ax1, session_data)
        
        # 2. Exercise breakdown
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_exercise_breakdown(ax2, session_data)
        
        # 3. Form score distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_form_distribution(ax3, session_data)
        
        # 4. Common mistakes heatmap
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_mistakes_heatmap(ax4, session_data)
        
        # 5. Progress over time
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_progress_timeline(ax5, session_data)
        
        plt.suptitle('Workout Performance Dashboard', fontsize=16, fontweight='bold')
        return fig
    
    def _plot_session_overview(self, ax, session_data: Dict):
        """Plot session overview metrics"""
        total_reps = session_data.get('total_reps', 0)
        avg_score = session_data.get('avg_form_score', 0)
        duration = session_data.get('duration_minutes', 0)
        
        metrics = ['Total Reps', 'Avg Score (%)', 'Duration (min)']
        values = [total_reps, avg_score, duration]
        
        bars = ax.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange'])
        ax.set_title('Session Overview')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}', ha='center', va='bottom')
    
    def _plot_exercise_breakdown(self, ax, session_data: Dict):
        """Plot exercise type breakdown"""
        exercises = session_data.get('exercises', {})
        if not exercises:
            ax.text(0.5, 0.5, 'No exercises completed', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Exercise Breakdown')
            return
        
        exercise_names = list(exercises.keys())
        rep_counts = [exercises[ex]['reps'] for ex in exercise_names]
        
        ax.pie(rep_counts, labels=exercise_names, autopct='%1.1f%%', startangle=90)
        ax.set_title('Exercise Breakdown')
    
    def _plot_form_distribution(self, ax, session_data: Dict):
        """Plot form score distribution"""
        exercises = session_data.get('exercises', {})
        all_scores = []
        
        for exercise_data in exercises.values():
            all_scores.extend(exercise_data.get('form_scores', []))
        
        if not all_scores:
            ax.text(0.5, 0.5, 'No form data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Form Score Distribution')
            return
        
        ax.hist(all_scores, bins=20, color='lightblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(all_scores), color='red', linestyle='--', label=f'Average: {np.mean(all_scores):.1f}%')
        ax.set_xlabel('Form Score (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Form Score Distribution')
        ax.legend()
    
    def _plot_mistakes_heatmap(self, ax, session_data: Dict):
        """Plot heatmap of common mistakes"""
        exercises = session_data.get('exercises', {})
        all_mistakes = {}
        
        for exercise_data in exercises.values():
            mistakes = exercise_data.get('common_mistakes', [])
            for mistake in mistakes:
                all_mistakes[mistake] = all_mistakes.get(mistake, 0) + 1
        
        if not all_mistakes:
            ax.text(0.5, 0.5, 'No mistakes tracked', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Common Mistakes')
            return
        
        mistakes = list(all_mistakes.keys())[:5]  # Top 5 mistakes
        counts = [all_mistakes[m] for m in mistakes]
        
        bars = ax.barh(mistakes, counts, color='salmon')
        ax.set_xlabel('Frequency')
        ax.set_title('Top Mistakes')
        
        # Wrap long mistake text
        ax.set_yticklabels([mistake[:30] + '...' if len(mistake) > 30 else mistake for mistake in mistakes])
    
    def _plot_progress_timeline(self, ax, session_data: Dict):
        """Plot progress timeline showing improvement over session"""
        exercises = session_data.get('exercises', {})
        
        if not exercises:
            ax.text(0.5, 0.5, 'No timeline data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Progress Timeline')
            return
        
        # Combine all form scores with timestamps
        all_data = []
        for exercise_name, exercise_data in exercises.items():
            form_scores = exercise_data.get('form_scores', [])
            for i, score in enumerate(form_scores):
                all_data.append({'time': i, 'score': score, 'exercise': exercise_name})
        
        if not all_data:
            ax.text(0.5, 0.5, 'No progress data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Progress Timeline')
            return
        
        # Plot timeline
        exercise_colors = {'squat': 'blue', 'pushup': 'red', 'downward_dog': 'green', 'warrior_pose': 'purple'}
        
        for exercise_name in set(d['exercise'] for d in all_data):
            exercise_data = [d for d in all_data if d['exercise'] == exercise_name]
            times = [d['time'] for d in exercise_data]
            scores = [d['score'] for d in exercise_data]
            color = exercise_colors.get(exercise_name, 'gray')
            ax.plot(times, scores, 'o-', color=color, label=exercise_name.title(), alpha=0.8)
        
        ax.set_xlabel('Time (reps)')
        ax.set_ylabel('Form Score (%)')
        ax.set_title('Progress Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    def draw_tutorial_overlay(self, frame: np.ndarray, exercise: str, step: int) -> np.ndarray:
        """Draw tutorial overlay with step-by-step instructions"""
        tutorials = {
            'squat': [
                "1. Stand with feet shoulder-width apart",
                "2. Lower by bending knees and hips",
                "3. Keep chest up and back straight",
                "4. Go down until thighs are parallel",
                "5. Push through heels to stand up"
            ],
            'pushup': [
                "1. Start in plank position",
                "2. Keep body in straight line",
                "3. Lower chest to ground", 
                "4. Push back up to start",
                "5. Maintain core engagement"
            ],
            'downward_dog': [
                "1. Start on hands and knees",
                "2. Tuck toes under",
                "3. Lift hips up and back",
                "4. Straighten arms and legs",
                "5. Form inverted V shape"
            ]
        }
        
        instructions = tutorials.get(exercise, ["Select an exercise to see tutorial"])
        current_step = min(step, len(instructions) - 1)
        
        # Draw tutorial box
        height, width = frame.shape[:2]
        box_height = 150
        box_width = 400
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (width - box_width - 20, 20), 
                     (width - 20, box_height + 20), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw instructions
        y_start = 50
        for i, instruction in enumerate(instructions):
            color = self.colors['excellent'] if i == current_step else self.colors['neutral']
            cv2.putText(frame, instruction, (width - box_width, y_start + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame