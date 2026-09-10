#!/usr/bin/env python3
"""
Exercise Analysis Rules for all 24 exercise types in the QEVD dataset.
Defines per-exercise static_constraints and dynamic_requirements.
"""

def get_exercise_analysis_rules(exercise_type: str) -> dict:
    """Get exercise-specific analysis rules for static and dynamic constraints"""
    
    rules = {
        'squat': {

            'static_constraints': {
                'pelvis_list': {
                    'priority': 'high',
                    'user_message': 'Keep your hips level'
                },
                'lumbar_extension': {
                    'priority': 'high',
                    'user_message': 'Keep your back straight throughout the movement'
                },
                'lumbar_bending': {
                    'priority': 'medium',
                    'user_message': 'Squat straight down'
                },
                'thorax_extension': {
                    'priority': 'medium',
                    'user_message': 'Keep your chest up'
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'high',
                    'user_message': 'Bend your hips more',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'high',
                    'user_message': 'Bend your hips more',
                    'over_message': 'Left hip flexion past target'
                },
                'knee_angle_r': {
                    'priority': 'high',
                    'user_message': 'Squat deeper',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'high',
                    'user_message': 'Squat deeper',
                    'over_message': 'Left knee bending past target'
                },
                'hip_adduction_r': {
                    'priority': 'high',
                    'user_message': 'Keep your right knee aligned over your toes',
                    'over_message': 'Right leg stance wider than target'
                },
                'hip_adduction_l': {
                    'priority': 'high',
                    'user_message': 'Keep your left knee aligned over your toes',
                    'over_message': 'Left leg stance wider than target'
                },
                'hip_rotation_r': {
                    'priority': 'medium',
                    'user_message': 'Avoid twisting at the right hip',
                    'over_message': 'Right hip rotation past target'
                },
                'hip_rotation_l': {
                    'priority': 'medium',
                    'user_message': 'Avoid twisting at the left hip',
                    'over_message': 'Left hip rotation past target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l',
                'pelvis_list', 'lumbar_extension', 'lumbar_bending', 'thorax_extension',
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l'
            ]
        },
        'pushup': {
            # Per paper Sec 3.4.2 supp: shoulder + elbow are dynamic (active
            # in the press), trunk + hip + knee are static (body stays
            # straight from head to heels).
            'static_constraints': {
                'hip_flexion_r': {
                    'priority': 'critical',
                    'user_message': 'Keep your body in a straight line from head to heels'
                },
                'hip_flexion_l': {
                    'priority': 'critical',
                    'user_message': 'Keep your body in a straight line from head to heels'
                },
                'lumbar_extension': {
                    'priority': 'critical',
                    'user_message': 'Engage your core to keep spine neutral'
                },
                'pelvis_list': {
                    'priority': 'high',
                    'user_message': 'Keep your hips level'
                },
            },
            'dynamic_requirements': {
                'elbow_flexion_r': {
                    'priority': 'high',
                    'user_message': 'Lower your chest closer to the ground',
                    'over_message': 'Right elbow bending past target'
                },
                'elbow_flexion_l': {
                    'priority': 'high',
                    'user_message': 'Lower your chest closer to the ground',
                    'over_message': 'Left elbow bending past target'
                },
                'shoulder_r_z': {
                    'priority': 'medium',
                    'user_message': 'Lower your right shoulder through the press',
                    'over_message': 'Right arm reach past target'
                },
                'shoulder_l_z': {
                    'priority': 'medium',
                    'user_message': 'Lower your left shoulder through the press',
                    'over_message': 'Left arm reach past target'
                },
                'shoulder_r_x': {
                    'priority': 'medium',
                    'user_message': 'Tuck your right elbow toward your ribs',
                    'over_message': 'Right arm range past target'
                },
                'shoulder_l_x': {
                    'priority': 'medium',
                    'user_message': 'Tuck your left elbow toward your ribs',
                    'over_message': 'Left arm range past target'
                },
                'shoulder_r_y': {
                    'priority': 'low',
                    'user_message': 'Control your right shoulder rotation',
                    'over_message': 'Right shoulder rotation past target'
                },
                'shoulder_l_y': {
                    'priority': 'low',
                    'user_message': 'Control your left shoulder rotation',
                    'over_message': 'Left shoulder rotation past target'
                },
            },
            'key_joints': [
                'elbow_flexion_r', 'elbow_flexion_l', 'shoulder_r_z', 'shoulder_l_z',
                'shoulder_r_x', 'shoulder_l_x', 'shoulder_r_y', 'shoulder_l_y',
                'hip_flexion_r', 'hip_flexion_l', 'lumbar_extension', 'pelvis_list'
            ]
        },
        'jumping_jack': {
            # Per paper Sec 3.4.2 supp: shoulder + hip are dynamic (full-body
            # motion), trunk is static.
            'static_constraints': {
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Land with hips level'
                },
                'lumbar_extension': {
                    'priority': 'medium',
                    'user_message': 'Keep your torso upright throughout the movement'
                },
                'lumbar_bending': {
                    'priority': 'medium',
                    'user_message': 'Jump straight up'
                },
                'thorax_extension': {
                    'priority': 'medium',
                    'user_message': 'Keep your chest up through the jump'
                },
            },
            'dynamic_requirements': {
                'shoulder_r_z': {
                    'priority': 'high',
                    'user_message': 'Raise your right arm higher',
                    'over_message': 'Right arm reach past target'
                },
                'shoulder_l_z': {
                    'priority': 'high',
                    'user_message': 'Raise your left arm higher',
                    'over_message': 'Left arm reach past target'
                },
                'shoulder_r_x': {
                    'priority': 'medium',
                    'user_message': 'Raise your right arm straight up the side, not forward',
                    'over_message': 'Right arm range past target'
                },
                'shoulder_l_x': {
                    'priority': 'medium',
                    'user_message': 'Raise your left arm straight up the side, not forward',
                    'over_message': 'Left arm range past target'
                },
                'hip_adduction_r': {
                    'priority': 'high', 'direction': 'min',  # SKEL R abduction → negative
                    'user_message': 'Jump your right foot wider out',
                    'over_message': 'Right leg stance wider than target'
                },
                'hip_adduction_l': {
                    'priority': 'high', 'direction': 'min',  # SKEL L abduction → negative
                    'user_message': 'Jump your left foot wider out',
                    'over_message': 'Left leg stance wider than target'
                },
                'hip_rotation_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right hip facing forward',
                    'over_message': 'Right hip rotation past target'
                },
                'hip_rotation_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left hip facing forward',
                    'over_message': 'Left hip rotation past target'
                },
            },
            'key_joints': [
                'shoulder_r_z', 'shoulder_l_z', 'shoulder_r_x', 'shoulder_l_x',
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l',
                'pelvis_list', 'lumbar_extension', 'lumbar_bending', 'thorax_extension'
            ]
        },
        'high_knees': {
            'static_constraints': {
                'lumbar_extension': {
                    'priority': 'high',
                    'user_message': 'Keep your torso upright'
                },
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Keep your hips level as you alternate knees'
                },
                'hip_adduction_r': {
                    'priority': 'medium',
                    'user_message': 'Drive your right knee straight up'
                },
                'hip_adduction_l': {
                    'priority': 'medium',
                    'user_message': 'Drive your left knee straight up'
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'critical',
                    'user_message': 'Lift your knees higher',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'critical',
                    'user_message': 'Lift your knees higher',
                    'over_message': 'Left hip flexion past target'
                },
                'knee_angle_r': {
                    'priority': 'high',
                    'user_message': 'Bend your right knee fully as it comes up',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'high',
                    'user_message': 'Bend your left knee fully as it comes up',
                    'over_message': 'Left knee bending past target'
                },
                'shoulder_r_x': {
                    'priority': 'medium',
                    'user_message': 'Pump your arms in time with your knees',
                    'over_message': 'Right arm range past target'
                },
                'shoulder_l_x': {
                    'priority': 'medium',
                    'user_message': 'Pump your arms in time with your knees',
                    'over_message': 'Left arm range past target'
                },
                'hip_rotation_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right hip facing forward',
                    'over_message': 'Right hip rotation past target'
                },
                'hip_rotation_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left hip facing forward',
                    'over_message': 'Left hip rotation past target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l',
                'shoulder_r_x', 'shoulder_l_x', 'lumbar_extension', 'pelvis_list',
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l'
            ]
        },
        'squat_jumps': {
            'static_constraints': {
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Land with both hips level'
                },
                'lumbar_extension': {
                    'priority': 'high',
                    'user_message': 'Keep your back straight when you land'
                },
                'lumbar_bending': {
                    'priority': 'medium',
                    'user_message': 'Land straight down'
                },
                'thorax_extension': {
                    'priority': 'medium',
                    'user_message': 'Stay tall through your upper back on landing'
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'critical',
                    'user_message': 'Squat deeper before jumping',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'critical',
                    'user_message': 'Squat deeper before jumping',
                    'over_message': 'Left hip flexion past target'
                },
                'knee_angle_r': {
                    'priority': 'high',
                    'user_message': 'Bend your knees more to generate more power',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'high',
                    'user_message': 'Bend your knees more to generate more power',
                    'over_message': 'Left knee bending past target'
                },
                'shoulder_r_x': {
                    'priority': 'medium',
                    'user_message': 'Swing your arms up to drive the jump',
                    'over_message': 'Right arm range past target'
                },
                'shoulder_l_x': {
                    'priority': 'medium',
                    'user_message': 'Swing your arms up to drive the jump',
                    'over_message': 'Left arm range past target'
                },
                'hip_adduction_r': {
                    'priority': 'high',
                    'user_message': 'Land with your right knee aligned over your toes',
                    'over_message': 'Right leg stance wider than target'
                },
                'hip_adduction_l': {
                    'priority': 'high',
                    'user_message': 'Land with your left knee aligned over your toes',
                    'over_message': 'Left leg stance wider than target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l',
                'shoulder_r_x', 'shoulder_l_x', 'pelvis_list', 'lumbar_extension',
                'lumbar_bending', 'thorax_extension', 'hip_adduction_r', 'hip_adduction_l'
            ]
        },
        'mountain_climbers': {
            'static_constraints': {
                'shoulder_r_z': {
                    'priority': 'medium',
                    'user_message': 'Keep your right shoulder stacked over your wrist'
                },
                'shoulder_l_z': {
                    'priority': 'medium',
                    'user_message': 'Keep your left shoulder stacked over your wrist'
                },
                'elbow_flexion_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right elbow steady to hold the plank'
                },
                'elbow_flexion_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left elbow steady to hold the plank'
                },
                'lumbar_extension': {
                    'priority': 'critical',
                    'user_message': "Keep your body in plank position"
                },
                'pelvis_list': {
                    'priority': 'high',
                    'user_message': 'Keep your hips level'
                },
                'shoulder_r_x': {
                    'priority': 'low',
                    'user_message': 'Keep your right arm steady underneath you'
                },
                'shoulder_l_x': {
                    'priority': 'low',
                    'user_message': 'Keep your left arm steady underneath you'
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'high',
                    'user_message': 'Drive your knees closer to your chest',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'high',
                    'user_message': 'Drive your knees closer to your chest',
                    'over_message': 'Left hip flexion past target'
                },
                'knee_angle_r': {
                    'priority': 'high',
                    'user_message': 'Fully bend your right knee on the drive',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'high',
                    'user_message': 'Fully bend your left knee on the drive',
                    'over_message': 'Left knee bending past target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l',
                'shoulder_r_z', 'shoulder_l_z', 'elbow_flexion_r', 'elbow_flexion_l',
                'lumbar_extension', 'pelvis_list', 'shoulder_r_x', 'shoulder_l_x'
            ]
        },
        'butt_kickers': {
            'static_constraints': {
                'lumbar_extension': {
                    'priority': 'high',
                    'user_message': 'Keep your torso upright'
                },
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Keep your hips level as you kick'
                },
                'hip_adduction_r': {
                    'priority': 'medium',
                    'user_message': 'Kick straight back, not out to the side'
                },
                'hip_adduction_l': {
                    'priority': 'medium',
                    'user_message': 'Kick straight back, not out to the side'
                },
            },
            'dynamic_requirements': {
                'knee_angle_r': {
                    'priority': 'critical',
                    'user_message': 'Kick your right heel higher',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'critical',
                    'user_message': 'Kick your left heel higher',
                    'over_message': 'Left knee bending past target'
                },
                'hip_flexion_r': {
                    'priority': 'medium',
                    'user_message': 'Let your right hip cycle with the kick',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'medium',
                    'user_message': 'Let your left hip cycle with the kick',
                    'over_message': 'Left hip flexion past target'
                },
                'shoulder_r_x': {
                    'priority': 'medium',
                    'user_message': 'Swing your arms in time with the kicks',
                    'over_message': 'Right arm range past target'
                },
                'shoulder_l_x': {
                    'priority': 'medium',
                    'user_message': 'Swing your arms in time with the kicks',
                    'over_message': 'Left arm range past target'
                },
                'hip_rotation_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right hip facing forward',
                    'over_message': 'Right hip rotation past target'
                },
                'hip_rotation_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left hip facing forward',
                    'over_message': 'Left hip rotation past target'
                },
            },
            'key_joints': [
                'knee_angle_r', 'knee_angle_l', 'hip_flexion_r', 'hip_flexion_l',
                'shoulder_r_x', 'shoulder_l_x', 'lumbar_extension', 'pelvis_list',
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l'
            ]
        },
        'walking_lunges': {
            'static_constraints': {
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Keep your hips level'
                },
                'lumbar_extension': {
                    'priority': 'high',
                    'user_message': "Keep your torso upright"
                },
                'lumbar_bending': {
                    'priority': 'medium',
                    'user_message': 'Lunge straight forward'
                },
                'thorax_extension': {
                    'priority': 'medium',
                    'user_message': 'Stay tall through your upper back as you sink'
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'high',
                    'user_message': 'Step deeper into your lunge',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'high',
                    'user_message': 'Step deeper into your lunge',
                    'over_message': 'Left hip flexion past target'
                },
                'knee_angle_r': {
                    'priority': 'high',
                    'user_message': 'Lower down more',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'high',
                    'user_message': 'Lower down more',
                    'over_message': 'Left knee bending past target'
                },
                'hip_adduction_r': {
                    'priority': 'high',
                    'user_message': 'Keep your right knee aligned with your toes',
                    'over_message': 'Right leg stance wider than target'
                },
                'hip_adduction_l': {
                    'priority': 'high',
                    'user_message': 'Keep your left knee aligned with your toes',
                    'over_message': 'Left leg stance wider than target'
                },
                'hip_rotation_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right hip square',
                    'over_message': 'Right hip rotation past target'
                },
                'hip_rotation_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left hip square',
                    'over_message': 'Left hip rotation past target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l',
                'pelvis_list', 'lumbar_extension', 'lumbar_bending', 'thorax_extension',
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l'
            ]
        },
        'plank_taps': {
            'static_constraints': {
                'hip_flexion_r': {
                    'priority': 'critical',
                    'user_message': "Don't let your hips rock side to side"
                },
                'hip_flexion_l': {
                    'priority': 'critical',
                    'user_message': "Don't let your hips rock side to side"
                },
                'lumbar_extension': {
                    'priority': 'critical',
                    'user_message': 'Keep your spine neutral in plank position'
                },
                'pelvis_list': {
                    'priority': 'critical',
                    'user_message': 'Keep your hip up when one hand lifts'
                },
                'hip_adduction_r': {
                    'priority': 'high',
                    'user_message': 'Spread your feet wider for a stable base'
                },
                'hip_adduction_l': {
                    'priority': 'high',
                    'user_message': 'Spread your feet wider for a stable base'
                },
            },
            'dynamic_requirements': {
                'shoulder_r_z': {
                    'priority': 'medium',
                    'user_message': 'Make controlled right-hand taps',
                    'over_message': 'Right arm reach past target'
                },
                'shoulder_l_z': {
                    'priority': 'medium',
                    'user_message': 'Make controlled left-hand taps',
                    'over_message': 'Left arm reach past target'
                },
                'shoulder_r_x': {
                    'priority': 'medium',
                    'user_message': 'Reach your right hand across to the opposite shoulder',
                    'over_message': 'Right arm range past target'
                },
                'shoulder_l_x': {
                    'priority': 'medium',
                    'user_message': 'Reach your left hand across to the opposite shoulder',
                    'over_message': 'Left arm range past target'
                },
                'elbow_flexion_r': {
                    'priority': 'low',
                    'user_message': 'Tap with a controlled right elbow',
                    'over_message': 'Right elbow bending past target'
                },
                'elbow_flexion_l': {
                    'priority': 'low',
                    'user_message': 'Tap with a controlled left elbow',
                    'over_message': 'Left elbow bending past target'
                },
            },
            'key_joints': [
                'shoulder_r_z', 'shoulder_l_z', 'shoulder_r_x', 'shoulder_l_x',
                'elbow_flexion_r', 'elbow_flexion_l', 'hip_flexion_r', 'hip_flexion_l',
                'lumbar_extension', 'pelvis_list', 'hip_adduction_r', 'hip_adduction_l'
            ]
        },
        'quick_feet': {
            'static_constraints': {
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Keep your hips level through quick steps'
                },
                'lumbar_extension': {
                    'priority': 'high',
                    'user_message': 'Keep your torso stable'
                },
                'hip_adduction_r': {
                    'priority': 'medium',
                    'user_message': 'Drive your right foot straight down, not out'
                },
                'hip_adduction_l': {
                    'priority': 'medium',
                    'user_message': 'Drive your left foot straight down, not out'
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'medium',
                    'user_message': 'Make quick, light steps',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'medium',
                    'user_message': 'Make quick, light steps',
                    'over_message': 'Left hip flexion past target'
                },
                'knee_angle_r': {
                    'priority': 'medium',
                    'user_message': 'Stay springy through your right knee',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'medium',
                    'user_message': 'Stay springy through your left knee',
                    'over_message': 'Left knee bending past target'
                },
                'shoulder_r_x': {
                    'priority': 'medium',
                    'user_message': 'Pump your right arm in time with your steps',
                    'over_message': 'Right arm range past target'
                },
                'shoulder_l_x': {
                    'priority': 'medium',
                    'user_message': 'Pump your left arm in time with your steps',
                    'over_message': 'Left arm range past target'
                },
                'hip_rotation_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right hip facing forward',
                    'over_message': 'Right hip rotation past target'
                },
                'hip_rotation_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left hip facing forward',
                    'over_message': 'Left hip rotation past target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l',
                'shoulder_r_x', 'shoulder_l_x', 'pelvis_list', 'lumbar_extension',
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l'
            ]
        },

        'air_jump_rope': {
            'static_constraints': {
                'shoulder_r_z': {
                    'priority': 'medium',
                    'user_message': 'Keep your right shoulder relaxed and quiet'
                },
                'shoulder_l_z': {
                    'priority': 'medium',
                    'user_message': 'Keep your left shoulder relaxed and quiet'
                },
                'shoulder_r_x': {
                    'priority': 'medium',
                    'user_message': 'Tuck your right elbow at your side'
                },
                'shoulder_l_x': {
                    'priority': 'medium',
                    'user_message': 'Tuck your left elbow at your side'
                },
                'hip_flexion_r': {
                    'priority': 'medium',
                    'user_message': "Stay tall"
                },
                'hip_flexion_l': {
                    'priority': 'medium',
                    'user_message': "Stay tall"
                },
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Stay level'
                },
                'lumbar_extension': {
                    'priority': 'medium',
                    'user_message': 'Keep your body upright while jumping'
                },
            },
            'dynamic_requirements': {
                'knee_angle_r': {
                    'priority': 'medium',
                    'user_message': 'Stay springy through your knees',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'medium',
                    'user_message': 'Stay springy through your knees',
                    'over_message': 'Left knee bending past target'
                },
                'elbow_flexion_r': {
                    'priority': 'medium',
                    'user_message': 'Keep elbows close and rotate wrists',
                    'over_message': 'Right elbow bending past target'
                },
                'elbow_flexion_l': {
                    'priority': 'medium',
                    'user_message': 'Keep elbows close and rotate wrists',
                    'over_message': 'Left elbow bending past target'
                },
            },
            'key_joints': [
                'knee_angle_r', 'knee_angle_l', 'elbow_flexion_r', 'elbow_flexion_l',
                'shoulder_r_z', 'shoulder_l_z', 'shoulder_r_x', 'shoulder_l_x',
                'hip_flexion_r', 'hip_flexion_l', 'pelvis_list', 'lumbar_extension'
            ]
        },
        'good_mornings': {
            'static_constraints': {
                'lumbar_extension': {
                    'priority': 'critical',
                    'user_message': "Keep your spine straight"
                },
                'knee_angle_r': {
                    'priority': 'high',
                    'user_message': 'Keep a slight bend in your right knee'
                },
                'knee_angle_l': {
                    'priority': 'high',
                    'user_message': 'Keep a slight bend in your left knee'
                },
                'lumbar_bending': {
                    'priority': 'medium',
                    'user_message': 'Hinge straight forward'
                },
                'thorax_extension': {
                    'priority': 'medium',
                    'user_message': 'Keep your chest open'
                },
                'hip_adduction_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right hip square'
                },
                'hip_adduction_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left hip square'
                },
                'hip_rotation_r': {
                    'priority': 'medium',
                    'user_message': 'No twisting at the right hip as you hinge'
                },
                'hip_rotation_l': {
                    'priority': 'medium',
                    'user_message': 'No twisting at the left hip as you hinge'
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'critical',
                    'user_message': 'Push your hips back and hinge forward',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'critical',
                    'user_message': 'Push your hips back and hinge forward',
                    'over_message': 'Left hip flexion past target'
                },
                'pelvis_tilt': {
                    'priority': 'high', 'direction': 'min',  # forward hinge → pelvis anteriorly rotates → SKEL value goes more negative
                    'user_message': 'Let your pelvis tilt naturally with the movement',
                    'over_message': 'Pelvic tilt past target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'lumbar_extension', 'pelvis_tilt',
                'knee_angle_r', 'knee_angle_l', 'lumbar_bending', 'thorax_extension',
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l'
            ]
        },
        'moving_plank': {
            'static_constraints': {
                'elbow_flexion_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right arm straight as you step'
                },
                'elbow_flexion_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left arm straight as you step'
                },
                'lumbar_extension': {
                    'priority': 'critical',
                    'user_message': 'Keep your body in a straight plank'
                },
                'hip_flexion_r': {
                    'priority': 'critical',
                    'user_message': "Don't let your hips sag or pike"
                },
                'hip_flexion_l': {
                    'priority': 'critical',
                    'user_message': "Don't let your hips sag or pike"
                },
                'pelvis_list': {
                    'priority': 'high',
                    'user_message': 'Keep your hips level as you travel sideways'
                },
                'shoulder_r_z': {
                    'priority': 'medium',
                    'user_message': 'Stack your right shoulder over your wrist'
                },
                'shoulder_l_z': {
                    'priority': 'medium',
                    'user_message': 'Stack your left shoulder over your wrist'
                },
            },
            'dynamic_requirements': {
                'shoulder_r_x': {
                    'priority': 'high',
                    'user_message': 'Step your right hand out wider',
                    'over_message': 'Right arm range past target'
                },
                'shoulder_l_x': {
                    'priority': 'high',
                    'user_message': 'Step your left hand out wider',
                    'over_message': 'Left arm range past target'
                },
                'hip_adduction_r': {
                    'priority': 'medium',
                    'user_message': 'Step your right foot to follow your hand',
                    'over_message': 'Right leg stance wider than target'
                },
                'hip_adduction_l': {
                    'priority': 'medium',
                    'user_message': 'Step your left foot to follow your hand',
                    'over_message': 'Left leg stance wider than target'
                },
            },
            'key_joints': [
                'shoulder_r_x', 'shoulder_l_x', 'hip_adduction_r', 'hip_adduction_l',
                'elbow_flexion_r', 'elbow_flexion_l', 'lumbar_extension', 'hip_flexion_r',
                'hip_flexion_l', 'pelvis_list', 'shoulder_r_z', 'shoulder_l_z'
            ]
        },
        'lunge_jumps': {
            'static_constraints': {
                'lumbar_extension': {
                    'priority': 'high',
                    'user_message': 'Keep your torso upright during the jump'
                },
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Land with hips level'
                },
                'lumbar_bending': {
                    'priority': 'medium',
                    'user_message': 'Land straight'
                },
                'thorax_extension': {
                    'priority': 'medium',
                    'user_message': 'Stay tall through your upper back on landing'
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'critical',
                    'user_message': 'Get into a deep lunge before jumping',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'critical',
                    'user_message': 'Get into a deep lunge before jumping',
                    'over_message': 'Left hip flexion past target'
                },
                'knee_angle_r': {
                    'priority': 'high',
                    'user_message': 'Bend both knees to 90 degrees',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'high',
                    'user_message': 'Bend both knees to 90 degrees',
                    'over_message': 'Left knee bending past target'
                },
                'hip_adduction_r': {
                    'priority': 'high',
                    'user_message': 'Land with your right knee aligned over your toes',
                    'over_message': 'Right leg stance wider than target'
                },
                'hip_adduction_l': {
                    'priority': 'high',
                    'user_message': 'Land with your left knee aligned over your toes',
                    'over_message': 'Left leg stance wider than target'
                },
                'hip_rotation_r': {
                    'priority': 'medium',
                    'user_message': 'Square your right hip on landing',
                    'over_message': 'Right hip rotation past target'
                },
                'hip_rotation_l': {
                    'priority': 'medium',
                    'user_message': 'Square your left hip on landing',
                    'over_message': 'Left hip rotation past target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l',
                'lumbar_extension', 'pelvis_list', 'lumbar_bending', 'thorax_extension',
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l'
            ]
        },
        'puddle_jumps': {
            'static_constraints': {
                'hip_flexion_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right hip neutral between jumps'
                },
                'hip_flexion_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left hip neutral between jumps'
                },
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Keep your hips level on landings'
                },
                'lumbar_extension': {
                    'priority': 'medium',
                    'user_message': 'Keep your body upright while jumping'
                },
                'hip_rotation_r': {
                    'priority': 'medium',
                    'user_message': 'No twisting at the right hip on landing'
                },
                'hip_rotation_l': {
                    'priority': 'medium',
                    'user_message': 'No twisting at the left hip on landing'
                },
                'shoulder_r_x': {
                    'priority': 'low',
                    'user_message': 'Keep your right arm steady for balance'
                },
                'shoulder_l_x': {
                    'priority': 'low',
                    'user_message': 'Keep your left arm steady for balance'
                },
            },
            'dynamic_requirements': {
                'hip_adduction_r': {
                    'priority': 'high', 'direction': 'min',
                    'user_message': 'Jump wider from side to side',
                    'over_message': 'Right leg stance wider than target'
                },
                'hip_adduction_l': {
                    'priority': 'high', 'direction': 'min',
                    'user_message': 'Jump wider from side to side',
                    'over_message': 'Left leg stance wider than target'
                },
                'knee_angle_r': {
                    'priority': 'medium',
                    'user_message': 'Bend your right knee to load for the jump',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'medium',
                    'user_message': 'Bend your left knee to load for the jump',
                    'over_message': 'Left knee bending past target'
                },
            },
            'key_joints': [
                'hip_adduction_r', 'hip_adduction_l', 'knee_angle_r', 'knee_angle_l',
                'hip_flexion_r', 'hip_flexion_l', 'pelvis_list', 'lumbar_extension',
                'hip_rotation_r', 'hip_rotation_l', 'shoulder_r_x', 'shoulder_l_x'
            ]
        },
        'floor_touches': {
            'static_constraints': {
                'knee_angle_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right leg as straight as comfortable'
                },
                'knee_angle_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left leg as straight as comfortable'
                },
                'hip_adduction_r': {
                    'priority': 'medium',
                    'user_message': 'Fold straight forward'
                },
                'hip_adduction_l': {
                    'priority': 'medium',
                    'user_message': 'Fold straight forward'
                },
                'lumbar_bending': {
                    'priority': 'medium',
                    'user_message': 'Fold straight forward'
                },
                'thorax_extension': {
                    'priority': 'medium',
                    'user_message': 'Let your chest descend evenly toward your knees'
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'critical',
                    'user_message': 'Bend forward from your hips to reach the floor',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'critical',
                    'user_message': 'Bend forward from your hips to reach the floor',
                    'over_message': 'Left hip flexion past target'
                },
                'lumbar_extension': {
                    'priority': 'high', 'direction': 'min',
                    'user_message': 'Let your spine round naturally',
                    'over_message': 'Spine flexion past target'
                },
                'pelvis_tilt': {
                    'priority': 'high', 'direction': 'min',
                    'user_message': 'Tilt your pelvis forward as you fold',
                    'over_message': 'Pelvic tilt past target'
                },
                'shoulder_r_z': {
                    'priority': 'medium',  # HSMR shoulder_z increases when arms reach DOWN (golden negated to match)
                    'user_message': 'Reach your right arm down toward the floor',
                    'over_message': 'Right arm reach past target'
                },
                'shoulder_l_z': {
                    'priority': 'medium',
                    'user_message': 'Reach your left arm down toward the floor',
                    'over_message': 'Left arm reach past target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'lumbar_extension', 'pelvis_tilt',
                'shoulder_r_z', 'shoulder_l_z', 'knee_angle_r', 'knee_angle_l',
                'hip_adduction_r', 'hip_adduction_l', 'lumbar_bending', 'thorax_extension'
            ]
        },
        'squat_kicks': {
            'static_constraints': {
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Keep your hips level through the kick'
                },
                'lumbar_extension': {
                    'priority': 'high',
                    'user_message': 'Keep your back straight throughout'
                },
                'lumbar_bending': {
                    'priority': 'medium',
                    'user_message': 'Kick straight ahead'
                },
                'thorax_extension': {
                    'priority': 'medium',
                    'user_message': 'Stay tall through your chest on the kick'
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'critical',
                    'user_message': 'Squat deep then kick high',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'critical',
                    'user_message': 'Squat deep then kick high',
                    'over_message': 'Left hip flexion past target'
                },
                'knee_angle_r': {
                    'priority': 'high',
                    'user_message': 'Fully extend your right leg in the kick',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'high',
                    'user_message': 'Fully extend your left leg in the kick',
                    'over_message': 'Left knee bending past target'
                },
                'hip_adduction_r': {
                    'priority': 'high',
                    'user_message': 'Keep your right knee aligned over your toes',
                    'over_message': 'Right leg stance wider than target'
                },
                'hip_adduction_l': {
                    'priority': 'high',
                    'user_message': 'Keep your left knee aligned over your toes',
                    'over_message': 'Left leg stance wider than target'
                },
                'hip_rotation_r': {
                    'priority': 'medium',
                    'user_message': 'Avoid twisting at the right hip on the kick',
                    'over_message': 'Right hip rotation past target'
                },
                'hip_rotation_l': {
                    'priority': 'medium',
                    'user_message': 'Avoid twisting at the left hip on the kick',
                    'over_message': 'Left hip rotation past target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l',
                'pelvis_list', 'lumbar_extension', 'lumbar_bending', 'thorax_extension',
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l'
            ]
        },
        'standing_kicks': {
            'static_constraints': {
                'lumbar_extension': {
                    'priority': 'high',
                    'user_message': 'Keep your torso upright and stable'
                },
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Keep your standing hip level'
                },
                'lumbar_bending': {
                    'priority': 'medium',
                    'user_message': 'No sideways lean as you kick'
                },
                'thorax_extension': {
                    'priority': 'low',
                    'user_message': 'Keep your chest tall through the kick'
                },
                'hip_adduction_r': {
                    'priority': 'medium',
                    'user_message': 'Kick straight ahead'
                },
                'hip_adduction_l': {
                    'priority': 'medium',
                    'user_message': 'Kick straight ahead'
                },
                'hip_rotation_r': {
                    'priority': 'medium',
                    'user_message': 'Square your right hip toward the target'
                },
                'hip_rotation_l': {
                    'priority': 'medium',
                    'user_message': 'Square your left hip toward the target'
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'critical',
                    'user_message': 'Kick your right leg up to waist height',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'critical',
                    'user_message': 'Kick your left leg up to waist height',
                    'over_message': 'Left hip flexion past target'
                },
                'knee_angle_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right kicking leg straight',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left kicking leg straight',
                    'over_message': 'Left knee bending past target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l',
                'lumbar_extension', 'pelvis_list', 'lumbar_bending', 'thorax_extension',
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l'
            ]
        },
        'boxing_squat_punches': {
            'static_constraints': {
                'lumbar_extension': {
                    'priority': 'high',
                    'user_message': 'Keep your back straight while squatting'
                },
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Keep your hips level while punching'
                },
                'shoulder_r_z': {
                    'priority': 'medium',
                    'user_message': "Punch level"
                },
                'shoulder_l_z': {
                    'priority': 'medium',
                    'user_message': "Punch level"
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'high',
                    'user_message': 'Squat deeper while punching',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'high',
                    'user_message': 'Squat deeper while punching',
                    'over_message': 'Left hip flexion past target'
                },
                'shoulder_r_x': {
                    'priority': 'high',
                    'user_message': 'Extend your right punch fully',
                    'over_message': 'Right arm range past target'
                },
                'shoulder_l_x': {
                    'priority': 'high',
                    'user_message': 'Extend your left punch fully',
                    'over_message': 'Left arm range past target'
                },
                'elbow_flexion_r': {
                    'priority': 'high',
                    'user_message': 'Fully extend your right arm in the punch',
                    'over_message': 'Right elbow bending past target'
                },
                'elbow_flexion_l': {
                    'priority': 'high',
                    'user_message': 'Fully extend your left arm in the punch',
                    'over_message': 'Left elbow bending past target'
                },
                'knee_angle_r': {
                    'priority': 'medium',
                    'user_message': 'Bend your right knee to load the squat',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'medium',
                    'user_message': 'Bend your left knee to load the squat',
                    'over_message': 'Left knee bending past target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'shoulder_r_x', 'shoulder_l_x',
                'elbow_flexion_r', 'elbow_flexion_l', 'knee_angle_r', 'knee_angle_l',
                'lumbar_extension', 'pelvis_list', 'shoulder_r_z', 'shoulder_l_z'
            ]
        },
        'deltoid_stretch': {
            'static_constraints': {
                'shoulder_r_y': {
                    'priority': 'medium',
                    'user_message': 'No right shoulder rotation'
                },
                'shoulder_l_y': {
                    'priority': 'medium',
                    'user_message': 'No left shoulder rotation'
                },
                'lumbar_extension': {
                    'priority': 'low',
                    'user_message': 'Keep your torso upright'
                },
                'pelvis_list': {
                    'priority': 'low',
                    'user_message': 'Stand level'
                },
                'thorax_extension': {
                    'priority': 'low',
                    'user_message': 'Stay tall through your upper back'
                },
                'lumbar_bending': {
                    'priority': 'low',
                    'user_message': 'No side lean while stretching'
                },
            },
            'dynamic_requirements': {
                'shoulder_r_x': {
                    'priority': 'critical',
                    'user_message': 'Pull your right arm across your body',
                    'over_message': 'Right arm range past target'
                },
                'shoulder_l_x': {
                    'priority': 'critical',
                    'user_message': 'Pull your left arm across your body',
                    'over_message': 'Left arm range past target'
                },
                'shoulder_r_z': {
                    'priority': 'high',
                    'user_message': 'Bring your right arm across your chest',
                    'over_message': 'Right arm reach past target'
                },
                'shoulder_l_z': {
                    'priority': 'high',
                    'user_message': 'Bring your left arm across your chest',
                    'over_message': 'Left arm reach past target'
                },
                'elbow_flexion_r': {
                    'priority': 'medium',
                    'user_message': 'Use your right arm to pull gently',
                    'over_message': 'Right elbow bending past target'
                },
                'elbow_flexion_l': {
                    'priority': 'medium',
                    'user_message': 'Use your left arm to pull gently',
                    'over_message': 'Left elbow bending past target'
                },
            },
            'key_joints': [
                'shoulder_r_x', 'shoulder_l_x', 'shoulder_r_z', 'shoulder_l_z',
                'elbow_flexion_r', 'elbow_flexion_l', 'shoulder_r_y', 'shoulder_l_y',
                'lumbar_extension', 'pelvis_list', 'thorax_extension', 'lumbar_bending'
            ]
        },
        'quad_stretch': {
            'static_constraints': {
                'pelvis_list': {
                    'priority': 'medium',
                    'user_message': 'Keep both hips level while balancing'
                },
                'lumbar_extension': {
                    'priority': 'medium',
                    'user_message': 'Keep your torso tall and upright'
                },
                'lumbar_bending': {
                    'priority': 'medium',
                    'user_message': 'Stay vertical'
                },
                'thorax_extension': {
                    'priority': 'medium',
                    'user_message': 'Stay tall through your chest while stretching'
                },
                'hip_adduction_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right knee pointing down, in line with the standing leg'
                },
                'hip_adduction_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left knee pointing down, in line with the standing leg'
                },
                'hip_rotation_r': {
                    'priority': 'medium',
                    'user_message': 'Avoid twisting at the right hip'
                },
                'hip_rotation_l': {
                    'priority': 'medium',
                    'user_message': 'Avoid twisting at the left hip'
                },
            },
            'dynamic_requirements': {
                'knee_angle_r': {
                    'priority': 'critical',
                    'user_message': 'Pull your right heel toward your glutes',
                    'over_message': 'Right knee bending past target'
                },
                'knee_angle_l': {
                    'priority': 'critical',
                    'user_message': 'Pull your left heel toward your glutes',
                    'over_message': 'Left knee bending past target'
                },
                'hip_flexion_r': {
                    'priority': 'high', 'direction': 'min',
                    'user_message': 'Push your right hip forward slightly',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'high', 'direction': 'min',
                    'user_message': 'Push your left hip forward slightly',
                    'over_message': 'Left hip flexion past target'
                },
            },
            'key_joints': [
                'knee_angle_r', 'knee_angle_l', 'hip_flexion_r', 'hip_flexion_l',
                'pelvis_list', 'lumbar_extension', 'lumbar_bending', 'thorax_extension',
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l'
            ]
        },
        'shoulder_gators': {
            'static_constraints': {
                'elbow_flexion_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right arm straight through the circle'
                },
                'elbow_flexion_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left arm straight through the circle'
                },
                'shoulder_r_y': {
                    'priority': 'low',
                    'user_message': 'No extra right shoulder rotation'
                },
                'shoulder_l_y': {
                    'priority': 'low',
                    'user_message': 'No extra left shoulder rotation'
                },
                'thorax_extension': {
                    'priority': 'low',
                    'user_message': 'Stay tall through your upper back'
                },
                'lumbar_bending': {
                    'priority': 'low',
                    'user_message': 'No side lean while circling'
                },
                'lumbar_extension': {
                    'priority': 'low',
                    'user_message': 'Keep your torso still'
                },
                'pelvis_list': {
                    'priority': 'low',
                    'user_message': 'Stand level'
                },
            },
            'dynamic_requirements': {
                'shoulder_r_z': {
                    'priority': 'critical',
                    'user_message': 'Make large circles with your right arm',
                    'over_message': 'Right arm reach past target'
                },
                'shoulder_l_z': {
                    'priority': 'critical',
                    'user_message': 'Make large circles with your left arm',
                    'over_message': 'Left arm reach past target'
                },
                'shoulder_r_x': {
                    'priority': 'high',
                    'user_message': 'Circle your right arm forward and back',
                    'over_message': 'Right arm range past target'
                },
                'shoulder_l_x': {
                    'priority': 'high',
                    'user_message': 'Circle your left arm forward and back',
                    'over_message': 'Left arm range past target'
                },
            },
            'key_joints': [
                'shoulder_r_z', 'shoulder_l_z', 'shoulder_r_x', 'shoulder_l_x',
                'elbow_flexion_r', 'elbow_flexion_l', 'shoulder_r_y', 'shoulder_l_y',
                'thorax_extension', 'lumbar_bending', 'lumbar_extension', 'pelvis_list'
            ]
        },
        'toe_touchers': {
            'static_constraints': {
                'knee_angle_r': {
                    'priority': 'high',
                    'user_message': 'Keep your right leg as straight as possible'
                },
                'knee_angle_l': {
                    'priority': 'high',
                    'user_message': 'Keep your left leg as straight as possible'
                },
                'hip_adduction_r': {
                    'priority': 'medium',
                    'user_message': 'Keep your right hip square as you fold'
                },
                'hip_adduction_l': {
                    'priority': 'medium',
                    'user_message': 'Keep your left hip square as you fold'
                },
                'lumbar_bending': {
                    'priority': 'medium',
                    'user_message': 'Fold straight forward'
                },
                'thorax_extension': {
                    'priority': 'medium',
                    'user_message': 'Round your upper back evenly into the fold'
                },
            },
            'dynamic_requirements': {
                'hip_flexion_r': {
                    'priority': 'critical',
                    'user_message': 'Fold forward from your hips to touch toes',
                    'over_message': 'Right hip flexion past target'
                },
                'hip_flexion_l': {
                    'priority': 'critical',
                    'user_message': 'Fold forward from your hips to touch toes',
                    'over_message': 'Left hip flexion past target'
                },
                'lumbar_extension': {
                    'priority': 'high', 'direction': 'min',
                    'user_message': 'Let your spine round to reach further',
                    'over_message': 'Spine flexion past target'
                },
                'pelvis_tilt': {
                    'priority': 'high', 'direction': 'min',
                    'user_message': 'Tilt your pelvis forward',
                    'over_message': 'Pelvic tilt past target'
                },
                'shoulder_r_z': {
                    'priority': 'medium',  # HSMR shoulder_z increases when arms reach DOWN (golden negated to match)
                    'user_message': 'Reach your right arm toward your toes',
                    'over_message': 'Right arm reach past target'
                },
                'shoulder_l_z': {
                    'priority': 'medium',
                    'user_message': 'Reach your left arm toward your toes',
                    'over_message': 'Left arm reach past target'
                },
            },
            'key_joints': [
                'hip_flexion_r', 'hip_flexion_l', 'lumbar_extension', 'pelvis_tilt',
                'shoulder_r_z', 'shoulder_l_z', 'knee_angle_r', 'knee_angle_l',
                'hip_adduction_r', 'hip_adduction_l', 'lumbar_bending', 'thorax_extension'
            ]
        }
    }
    
    # Handle variations in exercise names
    exercise_name_mapping = {
        'squats': 'squat',
        'pushups': 'pushup',
        'push-ups': 'pushup',
        'jumping_jacks': 'jumping_jack',
        'toe_touches': 'toe_touchers',
        'floor_touch': 'floor_touches',
    }
    
    # Check if exercise_type needs mapping
    if exercise_type in exercise_name_mapping:
        exercise_type = exercise_name_mapping[exercise_type]
    
    # Direct hit
    if exercise_type in rules:
        return rules[exercise_type]

    for suffix in ('_left', '_right'):
        if exercise_type.endswith(suffix):
            base = exercise_type[: -len(suffix)]
            if base in rules:
                return rules[base]
    return {}




