#!/usr/bin/env python3
"""
Exercise Cycle Matching Module
Handles matching of detected cycles with golden standard patterns
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy import interpolate as _itp
import warnings


def _classify_cycle_start(joint: np.ndarray) -> str:
    """Classify a joint cycle by its starting phase.
    """
    L = len(joint)
    if L < 4:
        return 'unknown'
    margin = max(2, L // 6)
    i_min = int(np.argmin(joint))
    i_max = int(np.argmax(joint))
    near_boundary = lambda i: i < margin or i >= L - margin
    if near_boundary(i_min) and not near_boundary(i_max):
        return 'valley_start'
    if near_boundary(i_max) and not near_boundary(i_min):
        return 'peak_start'
    return 'unknown'


def compute_phase_offset(user_joint: np.ndarray,
                         golden_joint: np.ndarray) -> Dict[str, float]:
    """Per-cycle phase offset of joint extremes after φ-alignment.
    """
    if len(user_joint) < 2 or len(golden_joint) < 2:
        return {}

    user_start = _classify_cycle_start(user_joint)
    golden_start = _classify_cycle_start(golden_joint)
    if (user_start == 'unknown' or golden_start == 'unknown'
            or user_start != golden_start):
        return {
            'start_mismatch': True,
            'user_start': user_start,
            'golden_start': golden_start,
        }

    phi = _itp.interp1d(
        np.arange(len(user_joint)), user_joint,
        kind='linear', fill_value='extrapolate',
    )
    aligned_user = phi(np.linspace(0, len(user_joint) - 1, len(golden_joint)))

    g_len = len(golden_joint)
    off_max = int(np.argmax(aligned_user)) - int(np.argmax(golden_joint))
    off_min = int(np.argmin(aligned_user)) - int(np.argmin(golden_joint))
    return {
        'start_mismatch': False,
        'user_start': user_start,
        'golden_start': golden_start,
        'phase_offset_max': off_max,
        'phase_offset_min': off_min,
        'phase_offset_max_norm': off_max / g_len,
        'phase_offset_min_norm': off_min / g_len,
        'golden_len': g_len,
    }


class CycleMatcher:
    """Matches detected exercise cycles with golden standard patterns"""
    
    def __init__(self, pose_param_names: List[str], similarity_threshold: float = 0.7):
        """
        Initialize cycle matcher
        
        Args:
            pose_param_names: List of pose parameter names
            similarity_threshold: Minimum similarity for valid match
        """
        self.pose_param_names = pose_param_names
        self.similarity_threshold = similarity_threshold

    def _get_key_joints(self, exercise_type: str) -> List:
        """Per-exercise focused joint set \\mathcal{J}^* (paper Sec 3.3).
        """
        from src.exercise_analysis_rules import get_exercise_analysis_rules
        rules = get_exercise_analysis_rules(exercise_type)
        if rules and isinstance(rules.get('key_joints'), list):
            return rules['key_joints']
        return []
    
    def match_cycle_to_golden(self, cycle_data: np.ndarray, golden_data: np.ndarray, 
                             exercise_type: str) -> Dict:
        """
        Match a detected cycle to golden standard
        """
        try:
            # Resample cycles to same length for comparison
            resampled_cycle = self._resample_cycle(cycle_data, len(golden_data))
            
            # Get key joints for this exercise
            # Use valid HSMR parameter indices as default (left_hip_x=3, right_hip_x=6, left_knee_x=12, right_knee_x=15)
            key_joints = self._get_key_joints(exercise_type)
            joint_indices = self._get_joint_indices(key_joints)
            
            # Calculate similarities for different aspects
            similarities = {
                'overall': self._calculate_overall_similarity(resampled_cycle, golden_data),
                'joint_specific': self._calculate_joint_similarities(
                    resampled_cycle, golden_data, joint_indices
                ),
                'temporal': self._calculate_temporal_similarity(resampled_cycle, golden_data),
                'amplitude': self._calculate_amplitude_similarity(
                    resampled_cycle, golden_data, joint_indices
                )
            }
            
            # Calculate weighted overall score
            overall_score = self._calculate_weighted_score(similarities)
            
            # Determine match quality
            match_quality = self._categorize_match_quality(overall_score)
            
            return {
                'similarity_score': overall_score,
                'match_quality': match_quality,
                'detailed_similarities': similarities,
                'key_joints_used': key_joints,
                'is_valid_match': overall_score >= self.similarity_threshold
            }
            
        except Exception as e:
            return {
                'error': f"Cycle matching failed: {str(e)}",
                'similarity_score': 0.0,
                'match_quality': 'error'
            }
    
    def match_half_cycles(self, half_cycle_1: np.ndarray, half_cycle_2: np.ndarray,
                         golden_half_1: np.ndarray, golden_half_2: np.ndarray,
                         exercise_type: str) -> Dict:
        """
        Match half cycles to golden standard halves
        """
        try:
            # Match first half
            match_1 = self.match_cycle_to_golden(half_cycle_1, golden_half_1, exercise_type)
            
            # Match second half
            match_2 = self.match_cycle_to_golden(half_cycle_2, golden_half_2, exercise_type)
            
            # Calculate combined score
            combined_score = (match_1['similarity_score'] + match_2['similarity_score']) / 2
            
            # Check if we should try swapped matching (for alternating exercises)
            if exercise_type in ['high_knees', 'butt_kickers', 'mountain_climbers', 'walking_lunges']:
                # Try matching with swapped halves
                match_1_swap = self.match_cycle_to_golden(half_cycle_1, golden_half_2, exercise_type)
                match_2_swap = self.match_cycle_to_golden(half_cycle_2, golden_half_1, exercise_type)
                combined_score_swap = (match_1_swap['similarity_score'] + match_2_swap['similarity_score']) / 2
                
                # Use the better matching
                if combined_score_swap > combined_score:
                    match_1, match_2 = match_1_swap, match_2_swap
                    combined_score = combined_score_swap
                    swapped = True
                else:
                    swapped = False
            else:
                swapped = False
            
            return {
                'half_1_match': match_1,
                'half_2_match': match_2,
                'combined_score': combined_score,
                'combined_quality': self._categorize_match_quality(combined_score),
                'halves_swapped': swapped,
                'is_valid_match': combined_score >= self.similarity_threshold
            }
            
        except Exception as e:
            return {
                'error': f"Half cycle matching failed: {str(e)}",
                'combined_score': 0.0,
                'combined_quality': 'error'
            }
    
    def find_best_match(self, cycles: List[np.ndarray], golden_data: np.ndarray,
                       exercise_type: str) -> Dict:
        """
        Find the best matching cycle from a list of detected cycles
        """
        if not cycles:
            return {
                'best_cycle_idx': -1,
                'best_score': 0.0,
                'best_match': None,
                'all_scores': []
            }
        
        matches = []
        scores = []
        
        for i, cycle in enumerate(cycles):
            match_result = self.match_cycle_to_golden(cycle, golden_data, exercise_type)
            matches.append(match_result)
            scores.append(match_result.get('similarity_score', 0.0))
        
        best_idx = np.argmax(scores)
        
        return {
            'best_cycle_idx': int(best_idx),
            'best_score': float(scores[best_idx]),
            'best_match': matches[best_idx],
            'all_scores': scores,
            'num_valid_matches': sum(1 for s in scores if s >= self.similarity_threshold)
        }
    
    def _resample_cycle(self, cycle_data: np.ndarray, target_length: int) -> np.ndarray:
        """Resample cycle data to target length"""
        
        if len(cycle_data) == target_length:
            return cycle_data
        
        # Use linear interpolation for resampling
        from scipy import interpolate
        
        original_indices = np.arange(len(cycle_data))
        target_indices = np.linspace(0, len(cycle_data) - 1, target_length)
        
        resampled = np.zeros((target_length, cycle_data.shape[1]))
        
        for j in range(cycle_data.shape[1]):
            f = interpolate.interp1d(original_indices, cycle_data[:, j], 
                                    kind='linear', fill_value='extrapolate')
            resampled[:, j] = f(target_indices)
        
        return resampled
    
    def _get_joint_indices(self, joint_names) -> List[int]:
        """Get indices for specified joint names or indices"""
        
        indices = []
        for joint in joint_names:
            if isinstance(joint, int):
                # Direct index - validate it's within bounds
                if 0 <= joint < 46:  # SMPL has 46 parameters
                    indices.append(joint)
                else:
                    warnings.warn(f"Parameter index {joint} out of bounds, skipping")
            elif isinstance(joint, str) and joint in self.pose_param_names:
                # Joint name
                indices.append(self.pose_param_names.index(joint))
            elif isinstance(joint, str):
                warnings.warn(f"Joint name '{joint}' not found in pose parameters")
        
        if not indices:
            # Fallback to first few parameters if no matches
            indices = list(range(min(4, len(self.pose_param_names))))
            warnings.warn("No matching joints found, using fallback indices")
        
        return indices
    
    def _calculate_overall_similarity(self, cycle1: np.ndarray, cycle2: np.ndarray) -> float:
        """Calculate overall similarity between two cycles"""
        
        try:
            # Flatten and compare
            flat1 = cycle1.flatten()
            flat2 = cycle2.flatten()
            
            # Use cosine similarity
            if np.linalg.norm(flat1) > 0 and np.linalg.norm(flat2) > 0:
                similarity = 1 - cosine(flat1, flat2)
            else:
                similarity = 0.0
            
            return float(np.clip(similarity, 0, 1))
            
        except:
            return 0.0
    
    def _calculate_joint_similarities(self, cycle1: np.ndarray, cycle2: np.ndarray,
                                    joint_indices: List[int]) -> Dict[str, float]:
        """Calculate similarity for each joint"""
        
        joint_sims = {}
        
        for idx in joint_indices:
            if idx < cycle1.shape[1] and idx < cycle2.shape[1]:
                joint1 = cycle1[:, idx]
                joint2 = cycle2[:, idx]
                
                if np.std(joint1) > 0 and np.std(joint2) > 0:
                    # Use correlation for joint similarity
                    corr, _ = pearsonr(joint1, joint2)
                    joint_sims[f'joint_{idx}'] = float(np.clip(corr, 0, 1))
                else:
                    joint_sims[f'joint_{idx}'] = 0.0
        
        return joint_sims
    
    def _calculate_temporal_similarity(self, cycle1: np.ndarray, cycle2: np.ndarray) -> float:
        """Calculate temporal pattern similarity"""
        
        try:
            # Compare derivatives (velocity patterns)
            diff1 = np.diff(cycle1, axis=0)
            diff2 = np.diff(cycle2, axis=0)
            
            if diff1.size > 0 and diff2.size > 0:
                flat_diff1 = diff1.flatten()
                flat_diff2 = diff2.flatten()
                
                if np.linalg.norm(flat_diff1) > 0 and np.linalg.norm(flat_diff2) > 0:
                    temporal_sim = 1 - cosine(flat_diff1, flat_diff2)
                    return float(np.clip(temporal_sim, 0, 1))
            
            return 0.5  # Neutral if can't calculate
            
        except:
            return 0.5
    
    def _calculate_amplitude_similarity(self, cycle1: np.ndarray, cycle2: np.ndarray,
                                      joint_indices: List[int]) -> float:
        """Calculate amplitude/range of motion similarity"""
        
        try:
            ranges1 = []
            ranges2 = []
            
            for idx in joint_indices:
                if idx < cycle1.shape[1] and idx < cycle2.shape[1]:
                    range1 = np.ptp(cycle1[:, idx])  # peak-to-peak
                    range2 = np.ptp(cycle2[:, idx])
                    
                    if range2 > 0:
                        # Calculate ratio (clamped to reasonable range)
                        ratio = np.clip(range1 / range2, 0.5, 2.0)
                        # Convert to similarity (1 when equal, lower when different)
                        similarity = 1 - abs(1 - ratio) / 1.5
                        ranges1.append(similarity)
            
            if ranges1:
                return float(np.mean(ranges1))
            
            return 0.5  # Neutral if can't calculate
            
        except:
            return 0.5
    
    def _calculate_weighted_score(self, similarities: Dict) -> float:
        """Calculate weighted overall score from different similarities"""
        
        # Weights for different similarity components
        weights = {
            'overall': 0.4,
            'joint_specific': 0.3,
            'temporal': 0.2,
            'amplitude': 0.1
        }
        
        score = 0.0
        
        # Overall similarity
        score += weights['overall'] * similarities.get('overall', 0)
        
        # Joint-specific similarities (average)
        joint_sims = similarities.get('joint_specific', {})
        if joint_sims:
            avg_joint_sim = np.mean(list(joint_sims.values()))
            score += weights['joint_specific'] * avg_joint_sim
        
        # Temporal similarity
        score += weights['temporal'] * similarities.get('temporal', 0.5)
        
        # Amplitude similarity
        score += weights['amplitude'] * similarities.get('amplitude', 0.5)
        
        return float(np.clip(score, 0, 1))
    
    
    def evaluate_by_extremes(self, cycle_data: np.ndarray, golden_data: np.ndarray,
                            exercise_type: str, feedback_timestamp: float = None,
                            cycle_start_timestamp: float = None,
                            video_ts_sec: np.ndarray = None) -> Dict:
        """
        Evaluate performance by comparing extreme values (min/max) instead of similarity
        Speed-invariant evaluation method with temporal extreme extraction
        """
        try:
            key_joints = self._get_key_joints(exercise_type)
            joint_indices = self._get_joint_indices(key_joints)

            evaluation = {
                'joint_evaluations': {},
                'overall_achievement': 0.0,
                'passed_requirements': 0,
                'total_requirements': 0,
                'details': {}
            }

            for i, idx in enumerate(joint_indices):
                joint_name = f'param_{idx}' if isinstance(key_joints[i], int) else key_joints[i]
                if idx < cycle_data.shape[1] and idx < golden_data.shape[1]:
                    user_joint = cycle_data[:, idx]
                    golden_joint = golden_data[:, idx]

                    # Calculate feedback frame within the cycle
                    feedback_frame = None
                    if feedback_timestamp is not None and cycle_start_timestamp is not None:
                        if video_ts_sec is not None:
                            # Accurate: use npy timestamps to find frame indices
                            fb_frame_abs = np.argmin(np.abs(video_ts_sec - feedback_timestamp))
                            cs_frame_abs = np.argmin(np.abs(video_ts_sec - cycle_start_timestamp))
                            feedback_frame = fb_frame_abs - cs_frame_abs
                        else:
                            # Fallback: old behavior (assumes 30fps, inaccurate with npy clock)
                            feedback_frame = int((feedback_timestamp - cycle_start_timestamp) * 30.0)
                        feedback_frame = max(0, min(len(user_joint) - 1, feedback_frame))
                    else:
                        feedback_frame = int(len(user_joint) * 0.8)

                    user_extremes = self._extract_temporal_extremes(user_joint, exercise_type, feedback_frame)
                    golden_extremes = self._extract_temporal_extremes(golden_joint, exercise_type)

                    if 'max' not in user_extremes:
                        user_extremes.update({
                            'max': float(np.max(user_joint)),
                            'min': float(np.min(user_joint)),
                            'range': float(np.ptp(user_joint)),
                            'max_frame': int(np.argmax(user_joint)),
                            'min_frame': int(np.argmin(user_joint))
                        })

                    if 'max' not in golden_extremes:
                        golden_extremes.update({
                            'max': float(np.max(golden_joint)),
                            'min': float(np.min(golden_joint)),
                            'range': float(np.ptp(golden_joint))
                        })

    
                    if len(user_joint) >= 2 and len(golden_joint) >= 2:
                        from scipy import interpolate as _itp
                        _phi = _itp.interp1d(
                            np.arange(len(user_joint)), user_joint,
                            kind='linear', fill_value='extrapolate'
                        )
                        _aligned_user = _phi(
                            np.linspace(0, len(user_joint) - 1, len(golden_joint))
                        )
                        _i_key_max = int(np.argmax(golden_joint))
                        _i_key_min = int(np.argmin(golden_joint))
                        user_extremes['temporal_at_golden_max'] = float(_aligned_user[_i_key_max])
                        user_extremes['temporal_at_golden_min'] = float(_aligned_user[_i_key_min])

                        _ttype = user_extremes.get('temporal_type', 'max')
                        user_extremes['temporal_extreme'] = (
                            user_extremes['temporal_at_golden_max'] if _ttype == 'max'
                            else user_extremes['temporal_at_golden_min']
                        )

                    joint_eval = {
                        'user_extremes': user_extremes,
                        'golden_extremes': golden_extremes,
                        'max_ratio': user_extremes['max'] / golden_extremes['max'] if golden_extremes['max'] != 0 else 0,
                        'range_ratio': user_extremes['range'] / golden_extremes['range'] if golden_extremes['range'] != 0 else 0,
                        'max_achieved': user_extremes['max'] >= golden_extremes['max'] * 0.85,
                        'range_achieved': user_extremes['range'] >= golden_extremes['range'] * 0.80
                    }

                    joint_score = 0.0
                    requirements = 0

                    if golden_extremes['max'] > 0.1:
                        joint_score += joint_eval['max_ratio']
                        requirements += 1

                    if golden_extremes['range'] > 0.1:
                        joint_score += joint_eval['range_ratio']
                        requirements += 1

                    if requirements > 0:
                        joint_eval['achievement_score'] = min(1.0, joint_score / requirements)
                    else:
                        joint_eval['achievement_score'] = 1.0

                    evaluation['joint_evaluations'][joint_name] = joint_eval
                    evaluation['total_requirements'] += requirements

                    if joint_eval.get('max_achieved', False):
                        evaluation['passed_requirements'] += 1
                    if joint_eval.get('range_achieved', False):
                        evaluation['passed_requirements'] += 1

            if evaluation['joint_evaluations']:
                achievement_scores = [j['achievement_score'] for j in evaluation['joint_evaluations'].values()]
                evaluation['overall_achievement'] = float(np.mean(achievement_scores))

            if evaluation['overall_achievement'] >= 0.9:
                evaluation['quality'] = 'excellent'
            elif evaluation['overall_achievement'] >= 0.75:
                evaluation['quality'] = 'good'
            elif evaluation['overall_achievement'] >= 0.6:
                evaluation['quality'] = 'acceptable'
            else:
                evaluation['quality'] = 'needs_improvement'

            return evaluation

        except Exception as e:
            return {
                'error': f"Extreme evaluation failed: {str(e)}",
                'overall_achievement': 0.0,
                'quality': 'error'
            }

    def _categorize_match_quality(self, score: float) -> str:
        """Categorize match quality based on score"""
        
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'good'
        elif score >= 0.7:
            return 'fair'
        elif score >= 0.6:
            return 'poor'
        else:
            return 'very_poor'
    
    def _extract_temporal_extremes(self, joint_data: np.ndarray, exercise_type: str, feedback_frame: int = None) -> Dict:
        """
        Extract extremes using temporal approach - finds extreme closest to feedback time
        If no feedback_frame provided, uses middle of cycle as reference
        """
        from scipy.signal import find_peaks
        
        if feedback_frame is None:
            # Use middle of cycle as default reference point
            feedback_frame = len(joint_data) // 2
        
        try:
            # Find all extremes based on exercise type
            if exercise_type in ['squat', 'high_knees', 'squat_jumps', 'lunge_jumps', 'standing_kicks', 'squat_kicks']:
                # Look for maxima (peaks) - higher values are better
                peaks, _ = find_peaks(joint_data, height=np.mean(joint_data), distance=5)
                extreme_frames = peaks
                extreme_values = joint_data[peaks]
                extreme_type = 'max'
                
            elif exercise_type in ['pushups', 'plank_taps', 'moving_plank']:
                # Look for minima (valleys) - lower values are better (closer to ground)
                valleys, _ = find_peaks(-joint_data, height=-np.mean(joint_data), distance=5)
                extreme_frames = valleys  
                extreme_values = joint_data[valleys]
                extreme_type = 'min'
                
            else:
                # Default to maxima for other exercises
                peaks, _ = find_peaks(joint_data, height=np.mean(joint_data), distance=5)
                extreme_frames = peaks
                extreme_values = joint_data[peaks]
                extreme_type = 'max'
            
            if len(extreme_frames) == 0:
                # Fallback to global extremes if no peaks/valleys found
                if extreme_type == 'min':
                    best_value = float(np.min(joint_data))
                    best_frame = int(np.argmin(joint_data))
                else:
                    best_value = float(np.max(joint_data))
                    best_frame = int(np.argmax(joint_data))
            else:
                # Find closest extreme to feedback time
                distances = [abs(frame - feedback_frame) for frame in extreme_frames]
                closest_idx = np.argmin(distances)
                best_value = float(extreme_values[closest_idx])
                best_frame = int(extreme_frames[closest_idx])
            
            # Return in format compatible with existing code
            result = {
                'max': best_value if extreme_type == 'max' else float(np.max(joint_data)),
                'min': best_value if extreme_type == 'min' else float(np.min(joint_data)),
                'range': float(np.ptp(joint_data)),
                'temporal_extreme': best_value,
                'temporal_frame': best_frame,
                'temporal_type': extreme_type
            }
            
            return result
            
        except Exception as e:
            # Fallback to simple approach
            warnings.warn(f"Temporal extreme extraction failed: {e}, using simple approach")
            return {
                'max': float(np.max(joint_data)),
                'min': float(np.min(joint_data)),
                'range': float(np.ptp(joint_data)),
                'max_frame': int(np.argmax(joint_data)),
                'min_frame': int(np.argmin(joint_data))
            }


def test_cycle_matcher():
    """Test the cycle matcher with sample data"""
    
    print("Testing Cycle Matcher...")
    
    # Create dummy pose parameter names
    pose_params = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 
                   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                   'torso_pitch', 'torso_roll', 'left_ankle', 'right_ankle',
                   'left_wrist', 'right_wrist'] + [f'param_{i}' for i in range(32)]
    
    matcher = CycleMatcher(pose_params)
    
    # Create sample cycles
    frames = 30
    t = np.linspace(0, 2*np.pi, frames)
    
    # Golden standard cycle
    golden = np.zeros((frames, 46))
    golden[:, 0] = np.sin(t)  # left_hip
    golden[:, 1] = np.sin(t + 0.1)  # right_hip
    golden[:, 2] = np.cos(t)  # left_knee
    golden[:, 3] = np.cos(t + 0.1)  # right_knee
    
    # Similar detected cycle (with some noise)
    detected_similar = golden + np.random.normal(0, 0.1, golden.shape)
    
    # Different detected cycle
    detected_different = np.zeros((frames, 46))
    detected_different[:, 0] = np.cos(2*t)
    detected_different[:, 1] = np.sin(3*t)
    
    # Test matching
    print("\nTest 1: Similar cycle matching")
    result1 = matcher.match_cycle_to_golden(detected_similar, golden, 'squat')
    print(f"  Similarity score: {result1['similarity_score']:.3f}")
    print(f"  Match quality: {result1['match_quality']}")
    print(f"  Is valid match: {result1['is_valid_match']}")
    
    print("\nTest 2: Different cycle matching")
    result2 = matcher.match_cycle_to_golden(detected_different, golden, 'squat')
    print(f"  Similarity score: {result2['similarity_score']:.3f}")
    print(f"  Match quality: {result2['match_quality']}")
    print(f"  Is valid match: {result2['is_valid_match']}")
    
    # Test half cycle matching
    print("\nTest 3: Half cycle matching")
    half1 = golden[:15]
    half2 = golden[15:]
    golden_half1 = golden[:15]
    golden_half2 = golden[15:]
    
    result3 = matcher.match_half_cycles(half1, half2, golden_half1, golden_half2, 'squat')
    print(f"  Combined score: {result3['combined_score']:.3f}")
    print(f"  Combined quality: {result3['combined_quality']}")
    print(f"  Halves swapped: {result3['halves_swapped']}")
    
    # Test finding best match
    print("\nTest 4: Finding best match from multiple cycles")
    cycles = [detected_different, detected_similar, golden + np.random.normal(0, 0.2, golden.shape)]
    result4 = matcher.find_best_match(cycles, golden, 'squat')
    print(f"  Best cycle index: {result4['best_cycle_idx']}")
    print(f"  Best score: {result4['best_score']:.3f}")
    print(f"  Valid matches: {result4['num_valid_matches']}")
    
    print("\n✅ Cycle Matcher test completed")


if __name__ == "__main__":
    test_cycle_matcher()