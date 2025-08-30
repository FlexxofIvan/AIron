import cv2
import numpy as np
from typing import Dict, List, Tuple
import time
import matplotlib.pyplot as plt

class TutorialMode:
    def __init__(self):
        self.current_step = 0
        self.step_start_time = time.time()
        self.step_duration = 5.0  # seconds per step
        self.tutorials = self._load_tutorials()
        
    def _load_tutorials(self) -> Dict:
        """Load tutorial data for each exercise"""
        return {
            'squats': {
                'steps': [
                    {
                        'instruction': "Stand with feet shoulder-width apart",
                        'check_points': ['feet_position'],
                        'duration': 5
                    },
                    {
                        'instruction': "Keep chest up and back straight",
                        'check_points': ['chest_up', 'back_straight'],
                        'duration': 3
                    },
                    {
                        'instruction': "Begin lowering by bending knees and hips",
                        'check_points': ['knee_bend', 'hip_bend'],
                        'duration': 4
                    },
                    {
                        'instruction': "Lower until thighs are parallel to ground",
                        'check_points': ['proper_depth'],
                        'duration': 3
                    },
                    {
                        'instruction': "Push through heels to return to start",
                        'check_points': ['heel_drive', 'full_extension'],
                        'duration': 3
                    }
                ]
            },
            'push-ups': {
                'steps': [
                    {
                        'instruction': "Start in plank position, hands under shoulders",
                        'check_points': ['hand_position', 'plank_form'],
                        'duration': 5
                    },
                    {
                        'instruction': "Keep body in straight line from head to heels",
                        'check_points': ['body_alignment'],
                        'duration': 4
                    },
                    {
                        'instruction': "Lower chest toward ground, elbows at 45°",
                        'check_points': ['elbow_angle', 'chest_depth'],
                        'duration': 3
                    },
                    {
                        'instruction': "Push back up to starting position",
                        'check_points': ['full_extension', 'control'],
                        'duration': 3
                    }
                ]
            },
            'downward dog': {
                'steps': [
                    {
                        'instruction': "Start on hands and knees",
                        'check_points': ['starting_position'],
                        'duration': 3
                    },
                    {
                        'instruction': "Tuck toes under and lift hips up",
                        'check_points': ['hip_lift'],
                        'duration': 4
                    },
                    {
                        'instruction': "Straighten arms and legs",
                        'check_points': ['arm_straight', 'leg_straight'],
                        'duration': 5
                    },
                    {
                        'instruction': "Form inverted V shape with body",
                        'check_points': ['triangle_formation'],
                        'duration': 4
                    }
                ]
            },
            'warrior pose': {
                'steps': [
                    {
                        'instruction': "Step left foot forward into lunge",
                        'check_points': ['foot_position'],
                        'duration': 4
                    },
                    {
                        'instruction': "Bend front knee to 90 degrees",
                        'check_points': ['knee_angle'],
                        'duration': 4
                    },
                    {
                        'instruction': "Raise arms overhead",
                        'check_points': ['arm_position'],
                        'duration': 3
                    },
                    {
                        'instruction': "Keep torso upright and strong",
                        'check_points': ['torso_alignment'],
                        'duration': 5
                    }
                ]
            }
        }
    
    def get_current_instruction(self, exercise: str) -> Dict:
        """Get current tutorial instruction"""
        exercise_key = exercise.lower().replace('-', '_').replace(' ', '_')
        tutorial = self.tutorials.get(exercise_key, {})
        
        if not tutorial or 'steps' not in tutorial:
            return {'instruction': 'Tutorial not available for this exercise', 'step': 0, 'total_steps': 0}
        
        steps = tutorial['steps']
        
        # Auto-advance steps based on time
        current_time = time.time()
        if current_time - self.step_start_time > self.step_duration:
            self.advance_step(len(steps))
        
        if self.current_step < len(steps):
            step_data = steps[self.current_step]
            return {
                'instruction': step_data['instruction'],
                'step': self.current_step + 1,
                'total_steps': len(steps),
                'check_points': step_data.get('check_points', []),
                'progress': min(100, ((current_time - self.step_start_time) / self.step_duration) * 100)
            }
        else:
            return {
                'instruction': 'Tutorial complete! Try the exercise.',
                'step': len(steps),
                'total_steps': len(steps),
                'check_points': [],
                'progress': 100
            }
    
    def advance_step(self, max_steps: int):
        """Advance to next tutorial step"""
        self.current_step = (self.current_step + 1) % max_steps
        self.step_start_time = time.time()
    
    def reset_tutorial(self):
        """Reset tutorial to beginning"""
        self.current_step = 0
        self.step_start_time = time.time()
    
    def check_step_completion(self, analysis_results: Dict, current_instruction: Dict) -> bool:
        """Check if current tutorial step is completed correctly"""
        check_points = current_instruction.get('check_points', [])
        
        if not check_points:
            return True
        
        # Check completion based on exercise analysis
        exercise = analysis_results.get('exercise', '')
        
        for check_point in check_points:
            if not self._evaluate_check_point(check_point, analysis_results):
                return False
        
        return True
    
    def _evaluate_check_point(self, check_point: str, analysis_results: Dict) -> bool:
        """Evaluate if a specific check point is met"""
        check_point_map = {
            'proper_depth': lambda r: r.get('proper_depth', False),
            'knee_alignment': lambda r: r.get('knee_alignment', False), 
            'back_straight': lambda r: r.get('back_straight', False),
            'body_alignment': lambda r: r.get('body_straight', False),
            'elbow_angle': lambda r: 70 < r.get('elbow_angle', 180) < 110,
            'full_extension': lambda r: r.get('elbow_angle', 0) > 160,
            'arm_straight': lambda r: r.get('arm_straight', False),
            'leg_straight': lambda r: r.get('leg_straight', False),
            'triangle_formation': lambda r: r.get('proper_triangle', False)
        }
        
        evaluator = check_point_map.get(check_point)
        if evaluator:
            return evaluator(analysis_results)
        
        return True  # Default to true for unknown check points

class ProgressVisualization:
    def __init__(self):
        self.session_history = []
        
    def create_progress_rings(self, current_reps: int, target_reps: int, form_score: float) -> np.ndarray:
        """Create circular progress visualization"""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        center = (150, 150)
        
        # Rep progress ring (outer)
        rep_angle = int((current_reps / target_reps) * 360) if target_reps > 0 else 0
        cv2.ellipse(img, center, (120, 120), 0, -90, -90 + rep_angle, (0, 255, 0), 15)
        cv2.ellipse(img, center, (120, 120), 0, -90 + rep_angle, 270, (50, 50, 50), 15)
        
        # Form score ring (inner)
        form_angle = int((form_score / 100) * 360)
        color = (0, 255, 0) if form_score >= 80 else (0, 255, 255) if form_score >= 60 else (0, 0, 255)
        cv2.ellipse(img, center, (80, 80), 0, -90, -90 + form_angle, color, 15)
        cv2.ellipse(img, center, (80, 80), 0, -90 + form_angle, 270, (30, 30, 30), 15)
        
        # Add text labels
        cv2.putText(img, f"Reps: {current_reps}/{target_reps}", (60, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Form: {form_score:.1f}%", (70, 170), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    def create_form_radar_chart(self, analysis_results: Dict) -> plt.Figure:
        """Create radar chart for different form aspects"""
        exercise = analysis_results.get('exercise', '')
        
        if exercise == 'squat':
            categories = ['Depth', 'Knee Alignment', 'Back Straight', 'Overall Form']
            values = [
                100 if analysis_results.get('proper_depth') else 0,
                100 if analysis_results.get('knee_alignment') else 0,
                100 if analysis_results.get('back_straight') else 0,
                analysis_results.get('form_score', 0)
            ]
        elif exercise == 'pushup':
            categories = ['Body Alignment', 'Elbow Angle', 'Range of Motion', 'Overall Form']
            values = [
                100 if analysis_results.get('body_straight') else 0,
                100 if analysis_results.get('at_bottom') else 0,
                100 if analysis_results.get('full_rom') else 0,
                analysis_results.get('form_score', 0)
            ]
        else:
            categories = ['Form Score']
            values = [analysis_results.get('form_score', 0)]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        values = values + [values[0]]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title(f'{exercise.title()} Form Analysis', pad=20)
        
        return fig