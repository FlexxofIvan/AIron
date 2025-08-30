import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List

class SessionTracker:
    def __init__(self):
        self.session_start = time.time()
        self.rep_count = 0
        self.form_scores = []
        self.exercise_type = None
        self.last_position = None
        self.rep_state = "up"  # "up" or "down" for rep counting
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'exercises': {},
            'total_reps': 0,
            'avg_form_score': 0
        }
        
    def update_session(self, analysis_results: Dict) -> Dict:
        """Update session with new analysis data"""
        if not analysis_results.get('detected', False):
            return self._get_session_stats()
        
        exercise = analysis_results.get('exercise')
        form_score = analysis_results.get('form_score', 0)
        
        if exercise and exercise != 'auto_detect':
            if exercise != self.exercise_type:
                self.exercise_type = exercise
                self.rep_count = 0
                self.rep_state = "up"
            
            # Add form score to tracking
            self.form_scores.append(form_score)
            
            # Rep counting logic
            self._count_reps(analysis_results)
            
            # Update session data
            if exercise not in self.session_data['exercises']:
                self.session_data['exercises'][exercise] = {
                    'reps': 0,
                    'form_scores': [],
                    'common_mistakes': []
                }
            
            self.session_data['exercises'][exercise]['reps'] = self.rep_count
            self.session_data['exercises'][exercise]['form_scores'].append(form_score)
            
            # Track common mistakes
            feedback = analysis_results.get('feedback', [])
            for mistake in feedback:
                if "excellent" not in mistake.lower() and "great" not in mistake.lower():
                    if mistake not in self.session_data['exercises'][exercise]['common_mistakes']:
                        self.session_data['exercises'][exercise]['common_mistakes'].append(mistake)
        
        return self._get_session_stats()
    
    def _count_reps(self, analysis_results: Dict):
        """Count repetitions based on exercise type"""
        exercise = analysis_results.get('exercise')
        
        if exercise == 'squat':
            self._count_squat_reps(analysis_results)
        elif exercise == 'pushup':
            self._count_pushup_reps(analysis_results)
    
    def _count_squat_reps(self, analysis_results: Dict):
        """Count squat repetitions"""
        knee_angle = analysis_results.get('knee_angle', 180)
        
        # State machine for rep counting
        if self.rep_state == "up" and knee_angle < 120:  # Going down
            self.rep_state = "down"
        elif self.rep_state == "down" and knee_angle > 150:  # Coming back up
            self.rep_state = "up"
            if analysis_results.get('proper_depth', False):  # Only count if good depth
                self.rep_count += 1
    
    def _count_pushup_reps(self, analysis_results: Dict):
        """Count push-up repetitions"""
        elbow_angle = analysis_results.get('elbow_angle', 180)
        
        # State machine for rep counting
        if self.rep_state == "up" and elbow_angle < 120:  # Going down
            self.rep_state = "down"
        elif self.rep_state == "down" and elbow_angle > 160:  # Coming back up
            self.rep_state = "up"
            if analysis_results.get('full_rom', False):  # Only count if full range
                self.rep_count += 1
    
    def _get_session_stats(self) -> Dict:
        """Get current session statistics"""
        session_duration = time.time() - self.session_start
        avg_form_score = np.mean(self.form_scores) if self.form_scores else 0
        
        return {
            'duration': f"{session_duration/60:.1f} min",
            'exercise': self.exercise_type or "None",
            'reps': self.rep_count,
            'avg_form_score': f"{avg_form_score:.1f}%",
            'current_form_scores': self.form_scores[-10:] if self.form_scores else []
        }
    
    def save_session(self, filename: str = None):
        """Save session data to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"
        
        self.session_data['end_time'] = datetime.now().isoformat()
        self.session_data['duration_minutes'] = (time.time() - self.session_start) / 60
        self.session_data['total_reps'] = self.rep_count
        self.session_data['avg_form_score'] = np.mean(self.form_scores) if self.form_scores else 0
        
        with open(filename, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        return filename
    
    def reset_session(self):
        """Reset session tracking"""
        self.session_start = time.time()
        self.rep_count = 0
        self.form_scores = []
        self.exercise_type = None
        self.rep_state = "up"
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'exercises': {},
            'total_reps': 0,
            'avg_form_score': 0
        }