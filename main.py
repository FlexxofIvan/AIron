import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pose_analyzer import PoseAnalyzer
from feedback_system import FeedbackSystem
from exercise_detector import ExerciseDetector
from session_tracker import SessionTracker
from visualizer import AdvancedVisualizer
from tutorial_mode import TutorialMode, ProgressVisualization
from advanced_features import WorkoutAnalytics, BiomechanicalAnalysis, MistakeHeatmap
from calibration import BodyCalibration, FormComparison

def main():
    st.set_page_config(
        page_title="Fitness Form Checker",
        page_icon="🏋️",
        layout="wide"
    )
    
    st.title("Real-Time Fitness Form Checker")
    
    if 'session_tracker' not in st.session_state:
        st.session_state.session_tracker = SessionTracker()
        st.session_state.pose_analyzer = PoseAnalyzer()
        st.session_state.feedback_system = FeedbackSystem()
        st.session_state.exercise_detector = ExerciseDetector()
        st.session_state.visualizer = AdvancedVisualizer()
        st.session_state.tutorial_mode_obj = TutorialMode()
        st.session_state.progress_viz = ProgressVisualization()
        st.session_state.analytics = WorkoutAnalytics()
        st.session_state.biomech_analysis = BiomechanicalAnalysis()
        st.session_state.mistake_heatmap = MistakeHeatmap()
        st.session_state.calibration = BodyCalibration()
        st.session_state.form_comparison = FormComparison()
        st.session_state.tutorial_enabled = False
        st.session_state.landmarks_history = []
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Live Feed", "Analytics", "3D View", "Tutorial"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            camera_placeholder = st.empty()
            
        with col2:
            st.subheader("Controls")
            exercise_mode = st.selectbox(
                "Exercise Mode",
                ["Auto-detect", "Squats", "Push-ups", "Downward Dog", "Warrior Pose"]
            )
            
            sensitivity = st.slider("Sensitivity", 0.5, 1.0, 0.8)
            voice_feedback = st.checkbox("Voice Feedback", True)
            show_angles = st.checkbox("Show Angles", True)
            show_zones = st.checkbox("Show Form Zones", True)
            show_3d = st.checkbox("Show 3D Pose", False)
            tutorial_enabled = st.checkbox("Tutorial Mode", False)
            target_reps = st.number_input("Target Reps", min_value=1, max_value=100, value=10)
            
            st.subheader("Calibration")
            if st.button("Calibrate Body Type"):
                st.session_state.calibration_mode = True
            
            if st.session_state.get('calibration_mode') and st.session_state.get('last_landmarks'):
                calib_result = st.session_state.calibration.calibrate_body_type(
                    st.session_state.last_landmarks, (480, 640)
                )
                if calib_result.get('calibrated'):
                    st.success(f"Calibrated for {calib_result['body_type']} body type")
                    st.session_state.calibration_mode = False
            
            st.subheader("Session Stats")
            stats_placeholder = st.empty()
            
            st.subheader("Form Feedback")
            feedback_placeholder = st.empty()
    
        run_camera = st.button("Start Camera")
        stop_camera = st.button("Stop Camera")
        
    with tab2:
        st.subheader("Workout Analytics")
        analytics_col1, analytics_col2 = st.columns(2)
        
        with analytics_col1:
            form_chart_placeholder = st.empty()
        
        with analytics_col2:
            progress_chart_placeholder = st.empty()
            
        dashboard_placeholder = st.empty()
        
        st.subheader("Advanced Analytics")
        radar_chart_placeholder = st.empty()
        biomech_placeholder = st.empty()
        
    with tab3:
        st.subheader("3D Pose Visualization")
        pose_3d_placeholder = st.empty()
        
    with tab4:
        st.subheader("Exercise Tutorial")
        tutorial_exercise = st.selectbox("Select Exercise for Tutorial", 
                                       ["Squats", "Push-ups", "Downward Dog", "Warrior Pose"])
        
        if st.button("Start Tutorial"):
            st.session_state.tutorial_enabled = True
            st.session_state.tutorial_mode_obj.reset_tutorial()
        
        if st.button("Next Step"):
            st.session_state.tutorial_mode_obj.advance_step(5)
        
        tutorial_placeholder = st.empty()
        tutorial_progress_placeholder = st.empty()
    
    if run_camera:
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, analysis_results = st.session_state.pose_analyzer.process_frame(
                frame, exercise_mode, sensitivity
            )
            
            # Store landmarks for advanced features
            if analysis_results.get('landmarks'):
                st.session_state.last_landmarks = analysis_results['landmarks']
                st.session_state.landmarks_history.append(analysis_results['landmarks'])
                if len(st.session_state.landmarks_history) > 100:
                    st.session_state.landmarks_history.pop(0)
            
            # Add tutorial overlay if enabled
            if tutorial_enabled:
                current_instruction = st.session_state.tutorial_mode_obj.get_current_instruction(exercise_mode)
                processed_frame = st.session_state.visualizer.draw_tutorial_overlay(
                    processed_frame, exercise_mode.lower().replace('-', '_'), current_instruction.get('step', 0)
                )
                
                # Update tutorial display
                tutorial_placeholder.write(f"**Step {current_instruction['step']}/{current_instruction['total_steps']}:** {current_instruction['instruction']}")
                progress_bar_value = current_instruction.get('progress', 0) / 100
                tutorial_progress_placeholder.progress(progress_bar_value)
            
            camera_placeholder.image(processed_frame, channels="BGR")
            
            feedback = st.session_state.feedback_system.generate_feedback(analysis_results)
            feedback_placeholder.write(feedback)
            
            stats = st.session_state.session_tracker.update_session(analysis_results)
            stats_placeholder.write(stats)
            
            # Update analytics visualizations
            if analysis_results.get('detected'):
                exercise = analysis_results.get('exercise', '')
                form_scores = st.session_state.session_tracker.form_scores
                
                # Form trend chart
                if form_scores:
                    fig_trend = st.session_state.visualizer.create_form_trend_chart(form_scores, exercise)
                    form_chart_placeholder.pyplot(fig_trend)
                    plt.close(fig_trend)
                
                # Progress visualization
                current_reps = st.session_state.session_tracker.rep_count
                fig_progress = st.session_state.visualizer.create_rep_progress_visualization(
                    current_reps, target_reps, form_scores[-current_reps:] if current_reps > 0 else []
                )
                progress_chart_placeholder.pyplot(fig_progress)
                plt.close(fig_progress)
                
                # 3D pose visualization
                if show_3d and analysis_results.get('landmarks'):
                    fig_3d = st.session_state.visualizer.create_3d_pose_visualization(
                        analysis_results['landmarks'], exercise
                    )
                    pose_3d_placeholder.pyplot(fig_3d)
                    plt.close(fig_3d)
                
                # Performance dashboard
                session_data = st.session_state.session_tracker.session_data
                if session_data.get('exercises'):
                    fig_dashboard = st.session_state.visualizer.create_performance_dashboard(session_data)
                    dashboard_placeholder.pyplot(fig_dashboard)
                    plt.close(fig_dashboard)
                
                # Radar chart for form analysis
                fig_radar = st.session_state.progress_viz.create_form_radar_chart(analysis_results)
                radar_chart_placeholder.pyplot(fig_radar)
                plt.close(fig_radar)
                
                # Biomechanical analysis
                if len(st.session_state.landmarks_history) >= 20:
                    biomech_results = st.session_state.biomech_analysis.analyze_movement_efficiency(
                        st.session_state.landmarks_history, exercise
                    )
                    if not biomech_results.get('insufficient_data'):
                        biomech_text = f"""
                        **Movement Analysis:**
                        - Smoothness: {biomech_results.get('movement_smoothness', 0):.1f}%
                        - Energy Efficiency: {biomech_results.get('energy_efficiency', 0):.1f}%
                        - Joint Stability: {list(biomech_results.get('joint_stability', {}).items())}
                        """
                        biomech_placeholder.markdown(biomech_text)
            
            if voice_feedback and feedback.get('voice_message'):
                st.session_state.feedback_system.speak(feedback['voice_message'])
        
        cap.release()

if __name__ == "__main__":
    main()