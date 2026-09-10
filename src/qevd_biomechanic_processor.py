#!/usr/bin/env python3
"""
QEVD Biomechanical Feedback Processor - Modular Version
Uses cycle_detection, cycle_matching, and exercise_analysis_rules modules
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import modular components
from src.cycle_detection import CycleDetector
from src.cycle_matching import CycleMatcher
from src.exercise_analysis_rules import get_exercise_analysis_rules


class QEVDBiomechanicProcessor:
    """QEVD Biomechanical Feedback Processor using modular architecture"""

    def __init__(self, golden_standards_dir: Optional[str] = None):
        # Golden-standard reference directory (configurable interface; defaults
        # to the repo-local ``golden_standards/``). Lets the YAML point it
        # elsewhere via ``golden_standards_dir`` without code changes.
        self.golden_standards_dir = (
            Path(golden_standards_dir) if golden_standards_dir
            else Path(__file__).resolve().parents[1] / 'golden_standards'
        )
        # Initialize pose parameters (46 HSMR parameters)
        self.pose_param_names = self._initialize_pose_params()

        # Initialize modular components
        self.cycle_detector = CycleDetector(self.pose_param_names)
        self.cycle_matcher = CycleMatcher(self.pose_param_names)

        # Exercise keyword mapping (for backward compatibility / normalization)
        self.exercise_keywords = {
            'squat': ['squat', 'squats'],
            'pushup': ['pushup', 'pushups', 'push up', 'push ups'],
            'jumping_jack': ['jumping jack', 'jumping jacks'],
            'high_knees': ['high knees', 'knees up', 'knees to waist', 'get those knees'],
            'squat_jumps': ['squat jump', 'squat jumps', 'jumping squat', 'jump squat'],
            'mountain_climbers': ['mountain climber', 'mountain climbers', 'mountain climbing'],
            'butt_kickers': ['butt kicker', 'butt kickers', 'heel kick', 'heel kicks'],
            'walking_lunges': ['walking lunge', 'walking lunges', 'lunge walk', 'forward lunge'],
            'plank_taps': ['plank tap', 'plank taps', 'shoulder tap', 'shoulder taps'],
            'quick_feet': ['quick feet', 'fast feet', 'quick step', 'rapid feet'],
            'air_jump_rope': ['air jump rope', 'jump rope', 'rope jumps'],
            'good_mornings': ['good morning', 'good mornings', 'good morning beginner'],
            'moving_plank': ['moving plank'],
            'lunge_jumps': ['lunge jumps', 'jumping lunges'],
            'puddle_jumps': ['puddle jumps'],
            'floor_touches': ['floor touches', 'toe touches', 'toe touchers'],
            'squat_kicks': ['squat kicks'],
            'standing_kicks': ['standing kicks'],
            'boxing_squat_punches': ['boxing squat punches', 'boxing punches'],
            'deltoid_stretch_left':  ['deltoid stretch left'],
            'deltoid_stretch_right': ['deltoid stretch right', 'deltoid stretch'],
            'quad_stretch_left':     ['quad stretch left'],
            'quad_stretch_right':    ['quad stretch right', 'quad stretch'],
            'shoulder_gators': ['shoulder gators'],
        }

    def _initialize_pose_params(self) -> List[str]:
        """Initialize pose parameter names for SMPL 46 parameters"""
        return [
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',  # 0-2
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',  # 3-5
            'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',  # 6-9
            'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',  # 10-12
            'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',  # 13-16
            'lumbar_bending', 'lumbar_extension', 'lumbar_twist',  # 17-19
            'thorax_bending', 'thorax_extension', 'thorax_twist',  # 20-22
            'head_bending', 'head_extension', 'head_twist',  # 23-25
            'scapula_r', 'scapula_elevation_r', 'scapula_upward_rot_r',  # 26-28
            'shoulder_r_x', 'shoulder_r_y', 'shoulder_r_z',  # 29-31
            'elbow_flexion_r', 'pronation_r', 'wrist_flexion_r', 'wrist_deviation_r',  # 32-35
            'scapula_l', 'scapula_elevation_l', 'scapula_upward_rot_l',  # 36-38
            'shoulder_l_x', 'shoulder_l_y', 'shoulder_l_z',  # 39-41
            'elbow_flexion_l', 'pronation_l', 'wrist_flexion_l', 'wrist_deviation_l'  # 42-45
        ]

    def _normalize_exercise_name(self, exercise_name: Optional[str]) -> Optional[str]:
        """Normalize exercise names to canonical forms used by analysis modules."""
        if not exercise_name:
            return None

        import re

        name = exercise_name.lower().strip()
        name = name.replace('-', ' ')
        name = re.sub(r'\s+', ' ', name)

        # Direct keyword mapping
        for canonical, keywords in self.exercise_keywords.items():
            if name in keywords:
                return canonical

        # Generic fallback: replace spaces with underscores
        canonical_candidate = name.replace(' ', '_')
        if canonical_candidate in self.exercise_keywords:
            return canonical_candidate

        # Special-case mappings
        special_map = {
            'good morning beginner': 'good_mornings',
            'good morning': 'good_mornings',
            'jump rope': 'air_jump_rope',
            'rope jumps': 'air_jump_rope',
            'quad stretch left': 'quad_stretch_left',
            'quad stretch right': 'quad_stretch_right',
            'quad stretch': 'quad_stretch_right',         # legacy / unspecified → right (matches original .npy)
            'deltoid stretch left': 'deltoid_stretch_left',
            'deltoid stretch right': 'deltoid_stretch_right',
            'deltoid stretch': 'deltoid_stretch_right',   # legacy / unspecified → right
            'armcrosschest left': 'deltoid_stretch_left',  # alias used in raw QEVD annotations
            'armcrosschest right': 'deltoid_stretch_right',
            'toe touchers': 'toe_touchers',
            'toe touches': 'toe_touchers',
        }
        if name in special_map:
            return special_map[name]

        return canonical_candidate if canonical_candidate else None

    def process_video_segment(self, skeleton_data: np.ndarray,
                             timestamps: np.ndarray,
                             exercise_name: str,
                             start_time: float,
                             end_time: float,
                             fps: float = 30.0,
                             video_ts_sec: np.ndarray = None,
                             exercise_start_time: float = None) -> Dict:
        """
        Process video segment and generate bio feedback

        Args:
            skeleton_data: HSMR pose data array (frames x 46 parameters)
            timestamps: Frame timestamps
            exercise_name: Type of exercise
            start_time: Segment start time (3s soft window for feedback)
            end_time: Segment end time (feedback moment)
            fps: Frames per second
            video_ts_sec: Per-frame npy timestamps (converted via /1e9 + 28800).
                          If provided, uses np.argmin for accurate timestamp-to-frame mapping.
            exercise_start_time: Absolute timestamp when the current exercise
                          episode started. Used to constrain cycle selection
                          to the current exercise (prevents picking cycles from
                          previous exercises in the same video). If None, no
                          constraint is applied (backward compatible).

        Returns:
            Dictionary with feedback information
        """

        # Normalize exercise name to canonical form used by analysis modules
        normalized_exercise = self._normalize_exercise_name(exercise_name)
        if normalized_exercise is None:
            return {"has_feedback": False, "error": f"Unsupported exercise: {exercise_name}"}

        # Extract pose matrix from HSMR data
        pose_matrix = self._extract_pose_matrix(skeleton_data)
        if pose_matrix is None:
            return {"has_feedback": False, "error": "Cannot extract pose data"}


        feedback_timestamp = end_time

        # Extract cycle before feedback point.
        version_a, version_b = self._extract_cycle_before_feedback(
            pose_matrix, normalized_exercise,
            feedback_timestamp=feedback_timestamp,
            exercise_start_time=exercise_start_time,
            fps=fps,
            video_ts_sec=video_ts_sec,
        )

        if version_a is None:
            return {"has_feedback": False, "error": "Cannot detect exercise cycle"}

        start_frame, end_frame = version_a

        # Get exercise analysis rules
        rules = get_exercise_analysis_rules(normalized_exercise)

        # Try to load golden standard
        golden_data = self._load_golden_standard(normalized_exercise)

        if golden_data is None:
            return {"has_feedback": False, "error": f"No golden standard for {exercise_name}"}

        # Extract cycle data
        cycle_data = pose_matrix[start_frame:end_frame]

        # Match cycle with golden standard using extreme-based evaluation
        match_result = self.cycle_matcher.match_cycle_to_golden(
            cycle_data, golden_data, normalized_exercise
        )

        if video_ts_sec is not None and 0 <= int(start_frame) < len(video_ts_sec):
            cycle_start_time = float(video_ts_sec[int(start_frame)])
        else:
            cycle_start_time = start_time + (start_frame / fps)
        extreme_eval = self.cycle_matcher.evaluate_by_extremes(
            cycle_data, golden_data, normalized_exercise,
            feedback_timestamp, cycle_start_time,
            video_ts_sec=video_ts_sec
        )

        # Generate feedback text based on evaluation
        feedback_text = self._generate_feedback_text(
            extreme_eval, rules, normalized_exercise, cycle_data
        )

        confidence = match_result.get('similarity_score', 0.0)

        return {
            "has_feedback": True,
            "feedback_text": feedback_text,
            "confidence": confidence,
            "detailed_analysis": {
                "match_result": match_result,
                "extreme_eval": extreme_eval
            },
            "cycle_info": {
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "matching_method": "extreme_evaluation"
            },
            "exercise_name": normalized_exercise,
        }

    def _extract_pose_matrix(self, hsmr_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose parameter matrix from HSMR data"""
        try:
            pose_matrices = []
            last_valid_pose = None

            for frame_data in hsmr_data:
                current_pose = None

                # Handle different HSMR data formats
                if isinstance(frame_data, dict) and 'poses' in frame_data:
                    poses = frame_data['poses']
                    if len(poses.shape) == 2 and poses.shape[0] > 0:
                        current_pose = poses[0][:46]  # Take first 46 params
                    elif len(poses.shape) == 1 and poses.shape[0] >= 46:
                        current_pose = poses[:46]
                elif isinstance(frame_data, np.ndarray) and len(frame_data) >= 46:
                    # Direct pose array
                    current_pose = frame_data[:46]

                if current_pose is not None:
                    pose_matrices.append(current_pose)
                    last_valid_pose = current_pose.copy()
                else:
                    if last_valid_pose is not None:
                        pose_matrices.append(last_valid_pose.copy())
                    else:
                        pose_matrices.append(np.zeros(46))

            if pose_matrices:
                return np.array(pose_matrices)
            return None

        except Exception:
            return None


    def _extract_cycle_before_feedback(self, pose_matrix: np.ndarray, exercise_type: str,
                                     feedback_timestamp: float, exercise_start_time: float,
                                     fps: float = 30.0, video_ts_sec: np.ndarray = None,
                                     ) -> Tuple:
        """
        Extract cycle before feedback point using modular cycle detector


        Returns:
            Tuple of (version_a, version_b)
            version_a: Complete previous cycle (start_frame, end_frame)
            version_b: From cycle start to feedback moment (start_frame, end_frame)
        """
        if video_ts_sec is not None:
            center_frame = np.argmin(np.abs(video_ts_sec - feedback_timestamp))
        else:
            center_frame = int(feedback_timestamp * fps)

        # Map exercise_start_time → frame index for cycle filtering.
        # A small buffer (-5 frames) tolerates cycle_detector boundary drift.
        if exercise_start_time is not None:
            if video_ts_sec is not None:
                ex_start_frame = int(np.argmin(np.abs(video_ts_sec - exercise_start_time))) - 5
            else:
                ex_start_frame = int(exercise_start_time * fps) - 5
            ex_start_frame = max(0, ex_start_frame)
        else:
            ex_start_frame = 0  # no constraint

        # Detect all cycles in the data
        detection_result = self.cycle_detector.detect_cycles(pose_matrix, exercise_type)

        if 'error' in detection_result:
            return None, None

        cycles = detection_result['cycles']

        if not cycles:
            # No cycles detected, use fixed window around feedback
            fixed_frames = 60  # Default 2 seconds
            start_frame = max(ex_start_frame, center_frame - fixed_frames // 2)
            end_frame = min(len(pose_matrix), start_frame + fixed_frames)
            return (start_frame, end_frame), None

        # Find the cycle that contains or is just before the feedback point
        # AND falls within the current exercise's time range.
        version_a = None
        version_b = None

        for cycle in reversed(cycles):
            cycle_start = cycle['start_frame']
            cycle_end = cycle['end_frame']

            # NEW: skip cycles that started before this exercise episode began
            if cycle_start < ex_start_frame:
                continue

            # Check if this cycle ends before or around the feedback point
            if cycle_end <= center_frame + 10:  # Allow small buffer
                # Version A: Complete previous cycle
                version_a = (cycle_start, cycle_end)

                # Version B: From cycle start to feedback moment
                if cycle_start < center_frame:
                    version_b = (cycle_start, center_frame)

                break

        # If no suitable cycle found in the current exercise, use the last cycle
        # WITHIN the exercise range (still filter ex_start_frame).
        if version_a is None and cycles:
            for cycle in reversed(cycles):
                if cycle['start_frame'] >= ex_start_frame and cycle['start_frame'] < center_frame:
                    version_a = (cycle['start_frame'], min(cycle['end_frame'], center_frame))
                    break

        return version_a, version_b

    def _load_golden_standard(self, exercise_name: str) -> Optional[np.ndarray]:
        """Load golden standard for an exercise (canonical name).

        All references live under project-local ``golden_standards/``; there
        is no longer any fallback to the shared ``/ceph/`` dataset.
        """
        # Simple aliases for legacy keys still appearing in some call sites
        name_mapping = {
            'pushups': 'pushup',
            'squats': 'squat',
            'jumping_jacks': 'jumping_jack',
            'good_morning_beginner': 'good_mornings',
            'good_mornings_beginner': 'good_mornings',
            # toe_touchers .npy was retired; floor_touches captures the same
            # forward-fold motion well enough per user's earlier call.
            'toe_touchers': 'floor_touches',
        }
        golden_exercise_name = name_mapping.get(exercise_name, exercise_name)

        golden_dir = self.golden_standards_dir
        path = golden_dir / f'realistic_golden_standard_{golden_exercise_name}.npy'
        if not path.exists():
            # Side-specific stretches only ship `_right`/`_left` NPYs; fall
            # back to the right variant when the base name is requested.
            alt = golden_dir / f'realistic_golden_standard_{golden_exercise_name}_right.npy'
            if alt.exists():
                path = alt
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                return None
        return None

    def _generate_feedback_text(self, extreme_eval: Dict, rules: Dict,
                                exercise_type: str, cycle_data: np.ndarray) -> str:
        """Generate feedback text based on extreme evaluation and rules.

        Picks up to 2 issues with this preference:
          1. one static + one dynamic (balanced form + range coverage)
          2. fall back to top-2 of whichever kind has violations
          3. "Good form!" if neither has violations
        Within each kind, ordering follows rule-dict insertion sequence,
        which is curated by per-exercise importance (matches key_joints
        ordering in ``exercise_analysis_rules``).
        """
        joint_evals = extreme_eval.get('joint_evaluations', {})

        static_issues: list = []
        dynamic_issues: list = []

        if cycle_data is not None and len(cycle_data) > 0:
            static_issues = self._check_static_constraints(
                cycle_data, rules.get('static_constraints', {})
            )
        if joint_evals:
            dynamic_issues = self._check_dynamic_requirements(
                joint_evals, rules.get('dynamic_requirements', {})
            )

        if static_issues and dynamic_issues:
            return f"{static_issues[0]}. Also, {dynamic_issues[0].lower()}"
        if static_issues:
            if len(static_issues) >= 2:
                return f"{static_issues[0]}. Also, {static_issues[1].lower()}"
            return static_issues[0]
        if dynamic_issues:
            if len(dynamic_issues) >= 2:
                return f"{dynamic_issues[0]}. Also, {dynamic_issues[1].lower()}"
            return dynamic_issues[0]
        return "Good form!"


    _WRAP_PRONE_PARAMS = (
        "pelvis_tilt", "pelvis_list", "pelvis_rotation",
        "lumbar_extension", "lumbar_bending", "lumbar_twist",
        "thorax_extension", "thorax_bending", "thorax_twist",
        "head_extension", "head_bending", "head_twist",
    )

    @staticmethod
    def _unwrap_then_fold(values_rad):
        """Return (continuous_unwrapped, folded_to_half_pi) for an HSMR Euler signal.


        """
        import numpy as _np
        arr = _np.asarray(values_rad, dtype=float)
        if arr.ndim == 0:
            return arr, ((arr + _np.pi / 2) % _np.pi) - _np.pi / 2
        unwrapped = _np.unwrap(arr, period=_np.pi)
        folded = ((unwrapped + _np.pi / 2) % _np.pi) - _np.pi / 2
        return unwrapped, folded


    @staticmethod
    def _humanize_joint(param_name: str) -> str:
        """Format a 46-dim pose parameter name for sentence-leading display.

        Examples:
            'hip_flexion_r' → 'Hip flexion (right)'
            'shoulder_r_x'  → 'Right shoulder X-axis'   (3-DoF joint: side in middle)
            'pelvis_tilt'   → 'Pelvis tilt'
        """
        import re
        # 3-DoF joints have the form <joint>_<r|l>_<x|y|z> (shoulder, scapula, etc.)
        m = re.match(r"^(.+?)_([rl])_([xyz])$", param_name)
        if m:
            joint, side, axis = m.groups()
            side_word = "right" if side == "r" else "left"
            return f"{side_word.capitalize()} {joint.replace('_', ' ')} {axis.upper()}-axis"
        # Single-DoF lateralized joints (hip_flexion_r, knee_angle_l, ...)
        if param_name.endswith("_r"):
            return (param_name[:-2].replace("_", " ") + " (right)").capitalize()
        if param_name.endswith("_l"):
            return (param_name[:-2].replace("_", " ") + " (left)").capitalize()
        # Mid-line joints (pelvis_tilt, lumbar_extension, ...)
        return param_name.replace("_", " ").capitalize()


    _STATIC_VAR_THRESHOLD_DEG = 5.0
    _DYNAMIC_TOLERANCE_DEG_DEFAULT = 10.0
    _DYNAMIC_TOLERANCE_DEG_CRITICAL = 5.0

    def _check_static_constraints(self, cycle_data: np.ndarray,
                                  static_constraints: Dict) -> List[str]:

        issues: List[str] = []
        for param_name, constraint in static_constraints.items():
            param_idx = self._get_param_index(param_name)
            if param_idx is None:
                continue
            if param_idx >= cycle_data.shape[1]:
                continue
            param_values = cycle_data[:, param_idx]
            if param_name in self._WRAP_PRONE_PARAMS:
                values_for_std, _ = self._unwrap_then_fold(param_values)
            else:
                values_for_std = param_values
            std_deg = float(np.degrees(np.std(values_for_std)))
            priority = constraint.get('priority', 'medium')
            if priority not in ['high', 'medium', 'critical']:
                continue
            if std_deg > self._STATIC_VAR_THRESHOLD_DEG:
                cue = constraint.get(
                    'user_message',
                    f'Stabilize {self._humanize_joint(param_name).lower()}',
                )
                evidence = (
                    f"{self._humanize_joint(param_name)} variance "
                    f"{std_deg:.0f}°, target < {self._STATIC_VAR_THRESHOLD_DEG:.0f}°"
                )
                issues.append(f"{evidence}. {cue}")
        return issues

    def _check_dynamic_requirements(self, joint_evals: Dict, dynamic_requirements: Dict) -> List[str]:

        issues = []

        for param_name, requirement in dynamic_requirements.items():
            param_index = self._get_param_index(param_name)
            if param_index is None:
                continue

            joint_eval = joint_evals.get(param_name)
            if joint_eval is None:
                joint_eval = joint_evals.get(f'param_{param_index}')
            if joint_eval is None:
                continue

            user_extremes = joint_eval.get('user_extremes', {})
            golden_extremes = joint_eval.get('golden_extremes', {})


            direction = requirement.get('direction', 'max')


            if direction == 'min':
                user_val = user_extremes.get('min', 0)
                golden_val = golden_extremes.get('min', 0)
            else:
                user_val = user_extremes.get('max', 0)
                golden_val = golden_extremes.get('max', 0)

            priority = requirement.get('priority', 'medium')
            if priority not in ['high', 'medium', 'critical']:
                continue
            tolerance = (self._DYNAMIC_TOLERANCE_DEG_CRITICAL if priority == 'critical'
                        else self._DYNAMIC_TOLERANCE_DEG_DEFAULT)


            if direction == 'min':
                undershoot_deg = float(np.degrees(user_val - golden_val))
            else:
                undershoot_deg = float(np.degrees(golden_val - user_val))

            if undershoot_deg > tolerance:
                # undershoot fires: user didn't reach reference range
                cue = requirement.get(
                    'user_message',
                    f'Adjust {self._humanize_joint(param_name).lower()} range',
                )
                deviation_deg = undershoot_deg
            elif -undershoot_deg > tolerance:
                # overshoot fires only if over_message provided
                over_cue = requirement.get('over_message')
                if not over_cue:
                    continue
                cue = over_cue
                deviation_deg = -undershoot_deg
            else:
                # within tolerance band: no issue
                continue

            evidence = (
                f"{self._humanize_joint(param_name)} "
                f"{np.degrees(user_val):.0f}°, target "
                f"{np.degrees(golden_val):.0f}° (within ±{tolerance:.0f}°)"
            )
            issues.append(f"{evidence}. {cue}")

        return issues

    def _get_param_index(self, param_name: str) -> Optional[int]:
        """Get parameter index from name"""
        if param_name in self.pose_param_names:
            return self.pose_param_names.index(param_name)
        return None
