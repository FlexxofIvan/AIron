#!/usr/bin/env python3
"""
Exercise Cycle Detection Module
Handles detection of exercise cycles from HSMR pose data
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, List, Tuple, Optional
import warnings

class CycleDetector:
    """Detects exercise cycles from pose data"""
    
    def __init__(self, pose_param_names: List[str], min_cycle_frames: int = 10, max_cycle_frames: int = 120):
        """
        Initialize cycle detector
        
        Args:
            pose_param_names: List of pose parameter names
            min_cycle_frames: Minimum frames for a valid cycle
            max_cycle_frames: Maximum frames for a valid cycle
        """
        self.pose_param_names = pose_param_names
        self.min_cycle_frames = min_cycle_frames
        self.max_cycle_frames = max_cycle_frames
        
        # Primary joints for each exercise (using SMPL parameter indices)
        # Based on movement variance analysis - using the first parameter with highest movement
        self.primary_joints = {
            'squat': 6,  # knee_angle_r
            'pushup': 32,  # elbow_flexion_r
            'jumping_jack': 31,  # shoulder_r_z
            'high_knees': 3,  # hip_flexion_r
            'squat_jumps': 6,  # knee_angle_r
            'mountain_climbers': 6,  # knee_angle_r
            'butt_kickers': 13,  # knee_angle_l
            'walking_lunges': 13,  # knee_angle_l
            'plank_taps': 31,  # shoulder_r_z
            'quick_feet': 13,  # knee_angle_l
            'air_jump_rope': 6,  # knee_angle_r
            'good_mornings': 10,  # hip_flexion_l
            'moving_plank': 29,  # shoulder_r_x
            'lunge_jumps': 13,  # knee_angle_l
            'puddle_jumps': 4,  # hip_adduction_r
            'floor_touches': 10,  # hip_flexion_l
            'squat_kicks': 10,  # hip_flexion_l
            'standing_kicks': 3,  # hip_flexion_r
            'boxing_squat_punches': 42,  # elbow_flexion_l
            'deltoid_stretch_right': 29,  # shoulder_r_x  (stretched arm pinned by L hand)
            'deltoid_stretch_left':  39,  # shoulder_l_x  (mirror)
            'quad_stretch_right': 6,   # knee_angle_r (heel→glute on R)
            'quad_stretch_left':  13,  # knee_angle_l (heel→glute on L)
            'shoulder_gators': 41,  # shoulder_l_z
            'toe_touchers': 10  # hip_flexion_l (dynamic bend, NOT static)
        }
    
    def detect_cycles(self, hsmr_data: np.ndarray, exercise_type: str) -> Dict:
        """
        Detect cycles in HSMR data for specific exercise
        
        Args:
            hsmr_data: HSMR pose data array
            exercise_type: Type of exercise
            
        Returns:
            Dictionary with detected cycles information
        """
        try:
            # Get primary joint for this exercise (parameter index)
            primary_joint_idx = self.primary_joints.get(exercise_type, 0)
            
            # Extract joint signal using parameter index
            if primary_joint_idx < hsmr_data.shape[1]:
                signal_data = hsmr_data[:, primary_joint_idx]
            else:
                # Fallback to first available joint
                signal_data = hsmr_data[:, 0]
                warnings.warn(f"Primary joint index {primary_joint_idx} out of bounds, using fallback")
            
            # Detect cycles based on exercise type
            if exercise_type in ['deltoid_stretch_left', 'deltoid_stretch_right',
                                 'quad_stretch_left',    'quad_stretch_right',
                                 'shoulder_gators']:
                # Static hold: user assumes a stretched pose for a long stable region
                cycles = self._detect_static_hold_cycles(signal_data)
            elif exercise_type in ['walking_lunges', 'lunge_jumps']:
                cycles = self._detect_alternating_cycles(signal_data)
            else:
                # toe_touchers is DYNAMIC (bend down → return up), uses periodic detection
                cycles = self._detect_periodic_cycles(signal_data)
            
            # Filter valid cycles
            valid_cycles = self._filter_valid_cycles(cycles, hsmr_data.shape[0])
            
            return {
                'cycles': valid_cycles,
                'num_cycles': len(valid_cycles),
                'primary_joint': primary_joint_idx,
                'signal_used': 'primary_joint_angle',
                'exercise_type': exercise_type
            }
            
        except Exception as e:
            return {
                'error': f"Cycle detection failed: {str(e)}",
                'cycles': [],
                'num_cycles': 0
            }
    
    def _detect_periodic_cycles(self, signal_data: np.ndarray) -> List[Tuple[int, int]]:
        """Detect periodic cycles using peak detection"""
        
        cycles = []
        
        # Smooth signal
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(signal_data, sigma=2)
        
        # Find peaks and valleys
        peaks, _ = scipy_signal.find_peaks(smoothed, prominence=0.1, distance=self.min_cycle_frames//2)
        valleys, _ = scipy_signal.find_peaks(-smoothed, prominence=0.1, distance=self.min_cycle_frames//2)
        
        # Create cycles from consecutive extrema
        all_extrema = sorted(list(peaks) + list(valleys))
        
        for i in range(len(all_extrema) - 1):
            start = all_extrema[i]
            end = all_extrema[i + 1]
            
            # Check if this could be a full cycle
            cycle_length = end - start
            if self.min_cycle_frames <= cycle_length <= self.max_cycle_frames:
                cycles.append((start, end))
        
        # Also try to detect cycles from peak to peak
        for i in range(len(peaks) - 1):
            start = peaks[i]
            end = peaks[i + 1]
            cycle_length = end - start
            if self.min_cycle_frames <= cycle_length <= self.max_cycle_frames:
                # Check if not already covered
                if not any(abs(c[0] - start) < 5 and abs(c[1] - end) < 5 for c in cycles):
                    cycles.append((start, end))
        
        return cycles
    
    def _detect_static_hold_cycles(self, signal_data: np.ndarray) -> List[Tuple[int, int]]:
        """Detect cycles for static hold exercises (stretches)"""
        
        cycles = []
        
        # For static holds, look for stable regions
        # Calculate rolling standard deviation
        window_size = 10
        rolling_std = np.array([
            np.std(signal_data[max(0, i-window_size):min(len(signal_data), i+window_size)])
            for i in range(len(signal_data))
        ])
        
        # Find stable regions (low std)
        threshold = np.percentile(rolling_std, 30)
        stable_mask = rolling_std < threshold
        
        # Find continuous stable regions
        in_stable = False
        start = 0
        
        for i, is_stable in enumerate(stable_mask):
            if is_stable and not in_stable:
                start = i
                in_stable = True
            elif not is_stable and in_stable:
                if i - start >= self.min_cycle_frames:
                    cycles.append((start, i))
                in_stable = False
        
        # Check last region
        if in_stable and len(signal_data) - start >= self.min_cycle_frames:
            cycles.append((start, len(signal_data)))
        
        return cycles
    
    def _detect_alternating_cycles(self, signal_data: np.ndarray) -> List[Tuple[int, int]]:
        """Detect cycles for alternating exercises (lunges, etc)"""
        
        cycles = []
        
        # For alternating movements, each leg movement is a cycle
        # Detect using zero crossings or mean crossings
        mean_val = np.mean(signal_data)
        
        # Find crossings
        crossings = np.where(np.diff(np.sign(signal_data - mean_val)))[0]
        
        # Create cycles from crossings
        for i in range(len(crossings) - 1):
            start = crossings[i]
            end = crossings[i + 1]
            cycle_length = end - start
            
            if self.min_cycle_frames <= cycle_length <= self.max_cycle_frames:
                cycles.append((start, end))
        
        return cycles
    
    def _filter_valid_cycles(self, cycles: List[Tuple[int, int]], data_length: int) -> List[Dict]:
        """Filter and format valid cycles"""
        
        valid_cycles = []
        
        for i, (start, end) in enumerate(cycles):
            # Ensure cycle is within bounds
            if start < 0 or end > data_length:
                continue
            
            # Ensure minimum length
            if end - start < self.min_cycle_frames:
                continue
            
            valid_cycles.append({
                'cycle_id': i,
                'start_frame': int(start),
                'end_frame': int(end),
                'duration': int(end - start),
                'type': 'full_cycle'
            })
        
        return valid_cycles
    
    def split_cycle_into_halves(self, cycle_data: np.ndarray, exercise_type: str) -> Dict:
        """
        Split a cycle into two half cycles
        
        Args:
            cycle_data: Data for one complete cycle
            exercise_type: Type of exercise
            
        Returns:
            Dictionary with two half cycles
        """
        try:
            primary_joint = self.primary_joints.get(exercise_type, 'left_hip')
            
            if primary_joint in self.pose_param_names:
                joint_idx = self.pose_param_names.index(primary_joint)
                signal_data = cycle_data[:, joint_idx]
            else:
                signal_data = cycle_data[:, 0]
            
            # Find split point
            split_point = self._find_cycle_split_point(signal_data, exercise_type)
            
            # Split the cycle
            half_1 = cycle_data[:split_point]
            half_2 = cycle_data[split_point:]
            
            return {
                'half_1': half_1,
                'half_2': half_2,
                'split_point': split_point,
                'half_1_frames': len(half_1),
                'half_2_frames': len(half_2),
                'descriptions': self._get_half_cycle_descriptions(exercise_type)
            }
            
        except Exception as e:
            return {
                'error': f"Cycle splitting failed: {str(e)}",
                'half_1': cycle_data[:len(cycle_data)//2],
                'half_2': cycle_data[len(cycle_data)//2:]
            }
    
    def _find_cycle_split_point(self, signal_data: np.ndarray, exercise_type: str) -> int:
        """Find the best point to split a cycle into halves"""
        
        # Find peaks and valleys
        peaks, _ = scipy_signal.find_peaks(signal_data, prominence=0.05)
        valleys, _ = scipy_signal.find_peaks(-signal_data, prominence=0.05)
        
        all_extrema = sorted(list(peaks) + list(valleys))
        
        if len(all_extrema) > 0:
            # Find extremum closest to center
            center = len(signal_data) // 2
            split_point = min(all_extrema, key=lambda x: abs(x - center))
        else:
            # Default to center if no extrema found
            split_point = len(signal_data) // 2
        
        # Ensure split point is reasonable
        min_half = len(signal_data) // 4
        max_half = 3 * len(signal_data) // 4
        split_point = max(min_half, min(split_point, max_half))
        
        return int(split_point)
    
    def _get_half_cycle_descriptions(self, exercise_type: str) -> Dict[str, str]:
        """Get descriptions for half cycles of each exercise"""
        
        descriptions = {
            'squat': {'half_1': 'descent phase', 'half_2': 'ascent phase'},
            'pushup': {'half_1': 'lowering phase', 'half_2': 'pushing phase'},
            'jumping_jack': {'half_1': 'opening phase', 'half_2': 'closing phase'},
            'high_knees': {'half_1': 'left knee up', 'half_2': 'right knee up'},
            'squat_jumps': {'half_1': 'squat phase', 'half_2': 'jump phase'},
            'mountain_climbers': {'half_1': 'left knee forward', 'half_2': 'right knee forward'},
            'butt_kickers': {'half_1': 'left heel kick', 'half_2': 'right heel kick'},
            'walking_lunges': {'half_1': 'lunge down', 'half_2': 'step forward'},
            'plank_taps': {'half_1': 'left hand tap', 'half_2': 'right hand tap'},
            'quick_feet': {'half_1': 'left foot up', 'half_2': 'right foot up'},
            'air_jump_rope': {'half_1': 'jump up', 'half_2': 'landing phase'},
            'good_mornings': {'half_1': 'bend forward', 'half_2': 'return upright'},
            'moving_plank': {'half_1': 'move left', 'half_2': 'move right'},
            'lunge_jumps': {'half_1': 'left leg forward', 'half_2': 'right leg forward'},
            'puddle_jumps': {'half_1': 'jump left', 'half_2': 'jump right'},
            'floor_touches': {'half_1': 'reach down', 'half_2': 'stand up'},
            'squat_kicks': {'half_1': 'squat down', 'half_2': 'kick forward'},
            'standing_kicks': {'half_1': 'left leg kick', 'half_2': 'right leg kick'},
            'boxing_squat_punches': {'half_1': 'squat down', 'half_2': 'punch up'},
            'deltoid_stretch': {'half_1': 'stretch hold', 'half_2': 'release'},
            'quad_stretch': {'half_1': 'stretch hold', 'half_2': 'release'},
            'shoulder_gators': {'half_1': 'rotate forward', 'half_2': 'rotate backward'},
            'toe_touchers': {'half_1': 'reach down', 'half_2': 'return up'}
        }
        
        return descriptions.get(exercise_type, {'half_1': 'first half', 'half_2': 'second half'})
    
    def calculate_cycle_statistics(self, cycles: List[Dict]) -> Dict:
        """Calculate statistics for detected cycles"""
        
        if not cycles:
            return {
                'num_cycles': 0,
                'avg_duration': 0,
                'min_duration': 0,
                'max_duration': 0,
                'total_frames': 0
            }
        
        durations = [c['duration'] for c in cycles]
        
        return {
            'num_cycles': len(cycles),
            'avg_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'total_frames': sum(durations),
            'consistency_score': 1.0 - (np.std(durations) / np.mean(durations)) if np.mean(durations) > 0 else 0
        }


def test_cycle_detector():
    """Test the cycle detector with sample data"""
    
    print("Testing Cycle Detector...")
    
    # Create dummy pose parameter names
    pose_params = [f'param_{i}' for i in range(46)]
    pose_params[0] = 'left_hip'
    pose_params[1] = 'left_shoulder'
    pose_params[2] = 'left_knee'
    
    detector = CycleDetector(pose_params)
    
    # Create sample sinusoidal data
    frames = 100
    t = np.linspace(0, 4*np.pi, frames)
    
    # Create mock HSMR data with periodic signal
    hsmr_data = np.zeros((frames, 46))
    hsmr_data[:, 0] = np.sin(t)  # left_hip
    hsmr_data[:, 1] = np.cos(t)  # left_shoulder
    hsmr_data[:, 2] = np.sin(2*t)  # left_knee
    
    # Test cycle detection for different exercises
    test_exercises = ['squat', 'pushup', 'jumping_jack', 'deltoid_stretch']
    
    for exercise in test_exercises:
        print(f"\nTesting {exercise}:")
        result = detector.detect_cycles(hsmr_data, exercise)
        
        if 'error' not in result:
            print(f"  Detected {result['num_cycles']} cycles")
            print(f"  Primary joint: {result['primary_joint']}")
            
            if result['cycles']:
                # Test splitting first cycle
                first_cycle = result['cycles'][0]
                cycle_data = hsmr_data[first_cycle['start_frame']:first_cycle['end_frame']]
                
                split_result = detector.split_cycle_into_halves(cycle_data, exercise)
                if 'error' not in split_result:
                    print(f"  Split cycle: {split_result['half_1_frames']} + {split_result['half_2_frames']} frames")
                    print(f"  Descriptions: {split_result['descriptions']}")
                
                # Calculate statistics
                stats = detector.calculate_cycle_statistics(result['cycles'])
                print(f"  Statistics: avg_duration={stats['avg_duration']:.1f}, consistency={stats['consistency_score']:.2f}")
        else:
            print(f"  Error: {result['error']}")
    
    print("\n✅ Cycle Detector test completed")


if __name__ == "__main__":
    test_cycle_detector()