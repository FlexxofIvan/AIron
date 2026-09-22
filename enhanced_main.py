import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pose_analyzer import PoseAnalyzer
from feedback_system import FeedbackSystem
from session_tracker import SessionTracker
from visualizer import AdvancedVisualizer
from tutorial_mode import TutorialMode, ProgressVisualization
from advanced_features import WorkoutAnalytics, BiomechanicalAnalysis, MistakeHeatmap
from calibration import BodyCalibration, FormComparison
import plotly.graph_objects as go
import plotly.express as px

def main():
    st.set_page_config(
        page_title="Advanced Fitness Form Checker",
        page_icon="🏋️‍♂️",
        layout="wide"
    )
    
    st.title("🏋️‍♂️ Advanced Real-Time Fitness Form Checker")
    st.markdown("---")
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.pose_analyzer = PoseAnalyzer()
        st.session_state.feedback_system = FeedbackSystem()
        st.session_state.session_tracker = SessionTracker()
        st.session_state.visualizer = AdvancedVisualizer()
        st.session_state.tutorial_mode = TutorialMode()
        st.session_state.progress_viz = ProgressVisualization()
        st.session_state.analytics = WorkoutAnalytics()
        st.session_state.biomech = BiomechanicalAnalysis()
        st.session_state.heatmap = MistakeHeatmap()
        st.session_state.calibration = BodyCalibration()
        st.session_state.comparison = FormComparison()
        st.session_state.landmarks_history = []
        st.session_state.camera_running = False
        st.session_state.initialized = True
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Exercise selection
        exercise_mode = st.selectbox(
            "Exercise Type",
            ["Auto-detect", "Squats", "Push-ups", "Downward Dog", "Warrior Pose"]
        )
        
        # Advanced settings
        st.subheader("⚙️ Settings")
        sensitivity = st.slider("Detection Sensitivity", 0.3, 1.0, 0.8, 0.1)
        target_reps = st.number_input("Target Reps", 1, 100, 10)
        
        # Visualization options
        st.subheader("👁️ Visualizations")
        show_skeleton = st.checkbox("Enhanced Skeleton", True)
        show_angles = st.checkbox("Angle Measurements", True)
        show_zones = st.checkbox("Form Zones", True)
        show_3d = st.checkbox("3D Pose View", False)
        
        # Feedback options
        st.subheader("🔊 Feedback")
        voice_feedback = st.checkbox("Voice Guidance", True)
        tutorial_mode = st.checkbox("Tutorial Mode", False)
        
        # Camera controls
        st.subheader("📹 Camera")
        col_start, col_stop = st.columns(2)
        with col_start:
            start_camera = st.button("▶️ Start", type="primary")
        with col_stop:
            stop_camera = st.button("⏹️ Stop")
            
        if stop_camera:
            st.session_state.camera_running = False
        
        # Calibration
        st.subheader("📏 Calibration")
        if st.button("🎯 Calibrate Body Type"):
            if hasattr(st.session_state, 'last_landmarks') and st.session_state.last_landmarks:
                calib_result = st.session_state.calibration.calibrate_body_type(
                    st.session_state.last_landmarks, (480, 640)
                )
                if calib_result.get('calibrated'):
                    st.success(f"✅ Calibrated: {calib_result['body_type'].replace('_', ' ').title()}")
                else:
                    st.error("❌ Calibration failed")
            else:
                st.warning("⚠️ Start camera first for calibration")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📹 Live Feed", "📊 Analytics", "🎯 3D View", "🎓 Tutorial", "📈 History"])
    
    with tab1:
        # Live video and real-time feedback
        col_video, col_feedback = st.columns([3, 1])
        
        with col_video:
            st.subheader("📹 Live Feed")
            video_placeholder = st.empty()
            
        with col_feedback:
            st.subheader("📋 Real-Time Feedback")
            feedback_placeholder = st.empty()
            
            st.subheader("📊 Session Stats")
            stats_placeholder = st.empty()
            
            st.subheader("🎯 Progress Ring")
            progress_ring_placeholder = st.empty()
    
    with tab2:
        st.subheader("📊 Workout Analytics")
        
        col_charts1, col_charts2 = st.columns(2)
        with col_charts1:
            st.write("**Form Score Trend**")
            trend_chart_placeholder = st.empty()
            
        with col_charts2:
            st.write("**Rep Progress**")
            rep_chart_placeholder = st.empty()
        
        st.write("**Performance Dashboard**")
        dashboard_placeholder = st.empty()
        
        col_radar, col_biomech = st.columns(2)
        with col_radar:
            st.write("**Form Radar Chart**")
            radar_placeholder = st.empty()
            
        with col_biomech:
            st.write("**Biomechanical Analysis**")
            biomech_placeholder = st.empty()
    
    with tab3:
        st.subheader("🎯 3D Pose Analysis")
        pose_3d_placeholder = st.empty()
        
        col_comparison, col_heatmap = st.columns(2)
        with col_comparison:
            st.write("**Form Comparison**")
            comparison_placeholder = st.empty()
            
        with col_heatmap:
            st.write("**Mistake Heatmap**")
            heatmap_placeholder = st.empty()
    
    with tab4:
        st.subheader("🎓 Exercise Tutorial")
        
        tutorial_exercise = st.selectbox(
            "Select Exercise",
            ["Squats", "Push-ups", "Downward Dog", "Warrior Pose"],
            key="tutorial_exercise"
        )
        
        col_tutorial_controls, col_tutorial_progress = st.columns(2)
        with col_tutorial_controls:
            if st.button("🎬 Start Tutorial"):
                st.session_state.tutorial_mode.reset_tutorial()
                st.success("Tutorial started!")
                
            if st.button("⏭️ Next Step"):
                st.session_state.tutorial_mode.advance_step(5)
        
        with col_tutorial_progress:
            tutorial_instruction_placeholder = st.empty()
            tutorial_progress_placeholder = st.empty()
    
    with tab5:
        st.subheader("📈 Workout History")
        history_placeholder = st.empty()
        
        if st.button("💾 Save Session"):
            filename = st.session_state.session_tracker.save_session()
            st.success(f"Session saved as {filename}")
        
        if st.button("🔄 Reset Session"):
            st.session_state.session_tracker.reset_session()
            st.session_state.landmarks_history = []
            st.success("Session reset!")
    
    # Camera processing loop
    if start_camera:
        st.session_state.camera_running = True
    
    if st.session_state.get('camera_running', False):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        while st.session_state.get('camera_running', False):
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Camera not available")
                break
            
            frame_count += 1
            
            # Process frame
            processed_frame, analysis_results = st.session_state.pose_analyzer.process_frame(
                frame, exercise_mode, sensitivity
            )
            
            # Store landmarks history
            if analysis_results.get('landmarks'):
                st.session_state.last_landmarks = analysis_results['landmarks']
                st.session_state.landmarks_history.append(analysis_results['landmarks'])
                if len(st.session_state.landmarks_history) > 100:
                    st.session_state.landmarks_history.pop(0)
            
            # Tutorial overlay
            if tutorial_mode:
                instruction_data = st.session_state.tutorial_mode.get_current_instruction(tutorial_exercise)
                processed_frame = st.session_state.visualizer.draw_tutorial_overlay(
                    processed_frame, tutorial_exercise.lower().replace('-', '_'), 
                    instruction_data.get('step', 0)
                )
                
                tutorial_instruction_placeholder.info(f"**Step {instruction_data['step']}/{instruction_data['total_steps']}:** {instruction_data['instruction']}")
                tutorial_progress_placeholder.progress(instruction_data.get('progress', 0) / 100)
            
            # Display processed video
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
            
            # Update feedback and stats
            feedback = st.session_state.feedback_system.generate_feedback(analysis_results)
            stats = st.session_state.session_tracker.update_session(analysis_results)
            
            feedback_placeholder.markdown(feedback.get('visual_feedback', 'Waiting for pose...'))
            stats_placeholder.json(stats)
            
            # Voice feedback
            if voice_feedback and feedback.get('voice_message'):
                st.session_state.feedback_system.speak(feedback['voice_message'])
            
            # Update visualizations (every 10 frames to reduce load)
            if frame_count % 10 == 0 and analysis_results.get('detected'):
                exercise = analysis_results.get('exercise', '')
                form_scores = st.session_state.session_tracker.form_scores
                current_reps = st.session_state.session_tracker.rep_count
                
                # Progress ring visualization
                if form_scores:
                    progress_ring = st.session_state.progress_viz.create_progress_rings(
                        current_reps, target_reps, form_scores[-1]
                    )
                    progress_ring_placeholder.image(progress_ring, channels="BGR")
                
                # Charts and analytics
                if form_scores and len(form_scores) > 5:
                    # Form trend
                    fig_trend = st.session_state.visualizer.create_form_trend_chart(form_scores, exercise)
                    trend_chart_placeholder.pyplot(fig_trend, use_container_width=True)
                    plt.close(fig_trend)
                    
                    # Rep progress
                    fig_progress = st.session_state.visualizer.create_rep_progress_visualization(
                        current_reps, target_reps, form_scores[-current_reps:] if current_reps > 0 else []
                    )
                    rep_chart_placeholder.pyplot(fig_progress, use_container_width=True)
                    plt.close(fig_progress)
                
                # 3D visualization
                if show_3d and analysis_results.get('landmarks'):
                    fig_3d = st.session_state.visualizer.create_3d_pose_visualization(
                        analysis_results['landmarks'], exercise
                    )
                    pose_3d_placeholder.pyplot(fig_3d, use_container_width=True)
                    plt.close(fig_3d)
                
                # Performance dashboard
                session_data = st.session_state.session_tracker.session_data
                if session_data.get('exercises'):
                    fig_dashboard = st.session_state.visualizer.create_performance_dashboard(session_data)
                    dashboard_placeholder.pyplot(fig_dashboard, use_container_width=True)
                    plt.close(fig_dashboard)
                
                # Radar chart
                if exercise != 'auto_detect':
                    fig_radar = st.session_state.progress_viz.create_form_radar_chart(analysis_results)
                    radar_placeholder.pyplot(fig_radar, use_container_width=True)
                    plt.close(fig_radar)
                
                # Biomechanical analysis
                if len(st.session_state.landmarks_history) >= 20:
                    biomech_results = st.session_state.biomech.analyze_movement_efficiency(
                        st.session_state.landmarks_history, exercise
                    )
                    if not biomech_results.get('insufficient_data'):
                        biomech_data = {
                            'Smoothness': biomech_results.get('movement_smoothness', 0),
                            'Energy Efficiency': biomech_results.get('energy_efficiency', 0),
                            'Tempo Score': biomech_results.get('tempo_analysis', {}).get('tempo_score', 0)
                        }
                        biomech_placeholder.json(biomech_data)
        
        cap.release()
        st.session_state.camera_running = False

if __name__ == "__main__":
    main()