import os
import sys
import warnings
import tempfile
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# 1. Подавление отладочных сообщений C++/TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_LOG_SEVERITY_THRESHOLD'] = '3'

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*SymbolDatabase.GetPrototype.*')

try:
    null_fds = [os.open(os.devnull, os.O_RDWR)]
    os.dup2(null_fds[0], 2)
except Exception:
    pass

# 2. Импорты компонентов системы
from pose_analyzer import PoseAnalyzer
from feedback_system import FeedbackSystem
from exercise_detector import ExerciseDetector
from session_tracker import SessionTracker
from visualizer import AdvancedVisualizer
from tutorial_mode import TutorialMode, ProgressVisualization
from advanced_features import WorkoutAnalytics, BiomechanicalAnalysis, MistakeHeatmap
from calibration import BodyCalibration, FormComparison
from video_processor import process_video_with_feedback


def normalize_exercise_key(exercise_name: str) -> str:
    """Приводит название упражнения к стандартизированному ключу для словарей."""
    key = exercise_name.lower().replace("-", "_").replace(" ", "_")
    if key in ["pushups", "pushup"]:
        return "push_ups"
    if key in ["squat"]:
        return "squats"
    if key in ["downwarddog"]:
        return "downward_dog"
    if key in ["warriorpose"]:
        return "warrior_pose"
    return key


def normalize_score(score: float) -> float:
    """
    Нормализует Form Score из диапазона [0, 100] в [-1, 1].
    Формула: (score - 50) / 50
    """
    if score is None:
        return 0.0
    # Ограничиваем входные значения границами [0, 100]
    clamped_score = max(0.0, min(100.0, float(score)))
    return (clamped_score - 50.0) / 50.0


def main():
    st.set_page_config(
        page_title="Fitness Form Checker",
        page_icon="🏋️",
        layout="wide"
    )

    st.title("Real-Time Fitness Form Checker")

    # Инициализация параметров сессии
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
        st.session_state.tutorial_step = 0
        st.session_state.landmarks_history = []
        st.session_state.camera_running = False
        st.session_state.last_landmarks = None
        st.session_state.last_exercise = "Auto-detect"

    # Создание вкладок
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Live Feed", 
        "Analytics", 
        "3D View", 
        "Tutorial", 
        "Video Upload & Analysis"
    ])

    # --- ВКЛАДКА 1: LIVE FEED ---
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
            tutorial_overlay = st.checkbox("Tutorial Mode Overlay", False)
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

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Start Camera"):
                st.session_state.camera_running = True
        with col_btn2:
            if st.button("Stop Camera"):
                st.session_state.camera_running = False

    # --- ВКЛАДКА 2: ANALYTICS ---
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

    # --- ВКЛАДКА 3: 3D POSE VISUALIZATION ---
    with tab3:
        st.subheader("3D Pose Visualization")
        
        col_left, col_center, col_right = st.columns([1, 2, 1])
        
        with col_center:
            pose_3d_placeholder = st.empty()
            
            if st.session_state.last_landmarks is not None:
                fig_3d = st.session_state.visualizer.create_3d_pose_visualization(
                    st.session_state.last_landmarks, 
                    st.session_state.last_exercise
                )
                fig_3d.set_size_inches(5, 4.5)
                pose_3d_placeholder.pyplot(fig_3d, use_container_width=True)
                plt.close(fig_3d)
            else:
                pose_3d_placeholder.info(
                    "No active pose detected. Please process a video in 'Video Upload & Analysis' tab or start the camera."
                )

    # --- ВКЛАДКА 4: EXERCISE TUTORIAL ---
    with tab4:
        st.subheader("Exercise Tutorial")
        
        tut_col1, tut_col2 = st.columns([1, 2])
        
        with tut_col1:
            tutorial_exercise = st.selectbox(
                "Select Exercise for Tutorial", 
                ["Squats", "Push-ups", "Downward Dog", "Warrior Pose"],
                key="tut_exercise_select"
            )

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Start Tutorial"):
                    st.session_state.tutorial_enabled = True
                    st.session_state.tutorial_mode_obj.reset_tutorial()

            with btn_col2:
                if st.button("Next Step"):
                    st.session_state.tutorial_mode_obj.advance_step(5)

        with tut_col2:
            tutorial_card = st.container()
            with tutorial_card:
                if st.session_state.get('tutorial_enabled', False):
                    normalized_key = normalize_exercise_key(tutorial_exercise)
                    current_instruction = st.session_state.tutorial_mode_obj.get_current_instruction(normalized_key)
                    
                    st.markdown(f"### Exercise: **{tutorial_exercise}**")
                    
                    if current_instruction.get('total_steps', 0) > 0:
                        st.markdown(f"#### Step {current_instruction['step']} / {current_instruction['total_steps']}")
                        st.info(f"👉 **Instruction:** {current_instruction['instruction']}")
                        
                        progress_val = min(1.0, max(0.0, current_instruction.get('progress', 0) / 100))
                        st.progress(progress_val)
                    else:
                        st.warning(f"Instruction not found for key '{normalized_key}'.")

                    st.markdown("""
                    **Tips for correct execution:**
                    * Keep your core engaged throughout the movement.
                    * Maintain steady breathing.
                    * Pay attention to joint alignment indicators in Live Feed.
                    """)
                else:
                    st.write("Click **Start Tutorial** to launch the step-by-step guidance.")

    # --- ВКЛАДКА 5: OBSЕРИВАНИЕ И СКАЧИВАНИЕ ВИДЕО ---
    with tab5:
        st.subheader("Upload & Analyze Video File")
        
        v_col1, v_col2 = st.columns([1, 1])

        with v_col1:
            video_exercise = st.selectbox(
                "Exercise Type for Video",
                ["Auto-detect", "Squats", "Push-ups", "Downward Dog", "Warrior Pose"],
                key="video_exercise_mode"
            )
            video_sensitivity = st.slider("Sensitivity", 0.5, 1.0, 0.8, key="video_sens")
            
            uploaded_file = st.file_uploader(
                "Choose a video file...", 
                type=["mp4", "mov", "avi"]
            )

        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            input_video_path = tfile.name

            output_video_path = input_video_path.replace(".mp4", "_processed.mp4")
            output_log_path = input_video_path.replace(".mp4", "_log.json")

            with v_col1:
                st.video(input_video_path)

            if st.button("Process Video"):
                with st.spinner("Analyzing video... Please wait"):
                    process_video_with_feedback(
                        input_path=input_video_path,
                        output_path=output_video_path,
                        log_path=output_log_path,
                        exercise_mode=video_exercise,
                        sensitivity=video_sensitivity
                    )
                st.success("Analysis complete!")

            if os.path.exists(output_video_path) and os.path.exists(output_log_path):
                with v_col2:
                    st.subheader("Processed Result")
                    st.video(output_video_path)
                    
                    st.write("---")
                    st.subheader("Download Files")
                    
                    with open(output_video_path, "rb") as vf:
                        st.download_button(
                            label="📥 Download Processed Video (.mp4)",
                            data=vf,
                            file_name="processed_workout.mp4",
                            mime="video/mp4"
                        )
                    
                    with open(output_log_path, "rb") as lf:
                        st.download_button(
                            label="📥 Download Analysis Log (.json)",
                            data=lf,
                            file_name="workout_feedback_log.json",
                            mime="application/json"
                        )

    # --- ЦИКЛ ОБРАБОТКИ С ВЕБ-КАМЕРЫ (LIVE FEED) ---
    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Unable to access the webcam. Please ensure a physical webcam is connected, or use the 'Video Upload & Analysis' tab.")
            st.session_state.camera_running = False
        else:
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from webcam.")
                    st.session_state.camera_running = False
                    break

                processed_frame, analysis_results = st.session_state.pose_analyzer.process_frame(
                    frame, exercise_mode, sensitivity
                )

                # Нормализация оценки формы (0..100 -> -1..1)
                if 'score' in analysis_results:
                    analysis_results['normalized_score'] = normalize_score(analysis_results['score'])

                if analysis_results.get('landmarks'):
                    st.session_state.last_landmarks = analysis_results['landmarks']
                    st.session_state.last_exercise = analysis_results.get('exercise', exercise_mode)
                    st.session_state.landmarks_history.append(analysis_results['landmarks'])
                    if len(st.session_state.landmarks_history) > 100:
                        st.session_state.landmarks_history.pop(0)

                if tutorial_overlay:
                    normalized_overlay_key = normalize_exercise_key(exercise_mode)
                    current_instruction = st.session_state.tutorial_mode_obj.get_current_instruction(normalized_overlay_key)
                    processed_frame = st.session_state.visualizer.draw_tutorial_overlay(
                        processed_frame, normalized_overlay_key, current_instruction.get('step', 0)
                    )

                camera_placeholder.image(processed_frame, channels="BGR")

                feedback = st.session_state.feedback_system.generate_feedback(analysis_results)
                feedback_placeholder.write(feedback)

                stats = st.session_state.session_tracker.update_session(analysis_results)
                
                # Добавляем отображение нормализованной оценки в интерфейс сессии
                if 'score' in analysis_results:
                    stats['Normalized Score'] = f"{analysis_results['normalized_score']:.2f}"
                stats_placeholder.write(stats)

                if analysis_results.get('detected'):
                    exercise = analysis_results.get('exercise', '')
                    raw_scores = st.session_state.session_tracker.form_scores

                    if raw_scores:
                        # Получаем список нормализованных оценок
                        normalized_scores = [normalize_score(s) for s in raw_scores]
                        
                        # График тренда можно строить как по нормализованным, так и по исходным данным
                        fig_trend = st.session_state.visualizer.create_form_trend_chart(raw_scores, exercise)
                        form_chart_placeholder.pyplot(fig_trend)
                        plt.close(fig_trend)

                    current_reps = st.session_state.session_tracker.rep_count
                    fig_progress = st.session_state.visualizer.create_rep_progress_visualization(
                        current_reps, target_reps, raw_scores[-current_reps:] if current_reps > 0 else []
                    )
                    progress_chart_placeholder.pyplot(fig_progress)
                    plt.close(fig_progress)

                    if show_3d and analysis_results.get('landmarks'):
                        fig_3d = st.session_state.visualizer.create_3d_pose_visualization(
                            analysis_results['landmarks'], exercise
                        )
                        fig_3d.set_size_inches(5, 4.5)
                        pose_3d_placeholder.pyplot(fig_3d, use_container_width=True)
                        plt.close(fig_3d)

                    if len(st.session_state.landmarks_history) >= 20:
                        biomech_results = st.session_state.biomech_analysis.analyze_movement_efficiency(
                            st.session_state.landmarks_history, exercise
                        )
                        if not biomech_results.get('insufficient_data'):
                            joint_stability = biomech_results.get('joint_stability', {})
                            
                            if isinstance(joint_stability, dict):
                                stability_items = [f"• **{k.capitalize()}**: {v:.1f}%" for k, v in joint_stability.items()]
                            elif isinstance(joint_stability, list):
                                stability_items = [f"• **{item[0].capitalize()}**: {item[1]:.1f}%" for item in joint_stability]
                            else:
                                stability_items = [str(joint_stability)]
                                
                            stability_str = "\n".join(stability_items)

                            biomech_text = f"""
                            **Movement Analysis:**
                            - **Smoothness:** {biomech_results.get('movement_smoothness', 0):.1f}%
                            - **Energy Efficiency:** {biomech_results.get('energy_efficiency', 0):.1f}%
                            
                            **Joint Stability:**
                            {stability_str}
                            """
                            biomech_placeholder.markdown(biomech_text)

                if voice_feedback and feedback.get('voice_message'):
                    st.session_state.feedback_system.speak(feedback['voice_message'])

            cap.release()


if __name__ == "__main__":
    main()