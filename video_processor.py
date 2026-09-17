import json
import logging
from typing import Dict, List, Tuple
import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:

  def __init__(self):
    self.mp_pose = mp.solutions.pose
    self.pose = self.mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    self.mp_drawing = mp.solutions.drawing_utils

  @staticmethod
  def calculate_angle(
      a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
  ) -> float:
    """Вычисление угла между двумя отрезками с вершиной в точке B (в градусах)."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    c_arr = np.array(c)

    radians = np.arctan2(c_arr[1] - b_arr[1], c_arr[0] - b_arr[0]) - np.arctan2(
        a_arr[1] - b_arr[1], a_arr[0] - b_arr[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
      angle = 360.0 - angle
    return angle

  def evaluate_frame(
      self, landmarks, exercise_mode: str, sensitivity: float
  ) -> Tuple[float, List[str]]:
    """Оценка техники по ключевым точкам для отдельного кадра."""
    feedback = []

    # Ключевые точки MediaPipe Pose:
    # 24: right_hip, 26: right_knee, 28: right_ankle
    hip = [landmarks[24].x, landmarks[24].y]
    knee = [landmarks[26].x, landmarks[26].y]
    ankle = [landmarks[28].x, landmarks[28].y]

    angle = self.calculate_angle(hip, knee, ankle)

    if exercise_mode == "squat":
      ideal_angle = 90.0
      angle_diff = abs(angle - ideal_angle)

      # Динамический расчет балла от 0 до 100
      penalty = angle_diff * (sensitivity * 1.5)
      score = max(0.0, min(100.0, 100.0 - penalty))

      if angle > 140:
        feedback.append("Опускайтесь ниже для полной амплитуды (squat)")
      elif angle < 70:
        feedback.append("Слишком глубокий присед, берегите колени")
      else:
        feedback.append("Отличная глубина приседа!")

    elif exercise_mode == "pushup":
      ideal_angle = 90.0
      angle_diff = abs(angle - ideal_angle)

      penalty = angle_diff * (sensitivity * 1.5)
      score = max(0.0, min(100.0, 100.0 - penalty))

      if angle > 130:
        feedback.append("Опускайтесь ниже при отжимании (pushup)")
      else:
        feedback.append("Хорошая амплитуда отжимания!")

    else:
      score = 50.0
      feedback.append(f"Неизвестный режим: {exercise_mode}")

    return round(score, 2), feedback

  def process_file(
      self,
      input_path: str,
      output_video_path: str,
      output_path: str,
      log_path: str,
      exercise_mode: str,
      sensitivity: float,
  ):
    """Основной метод обработки видео с генерацией размеченного файла и JSON-лога."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
      raise ValueError(f"Не удалось открыть видеофайл: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_logs = []
    scores_list = []
    frame_count = 0

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      frame_count += 1
      timestamp = frame_count / fps

      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = self.pose.process(rgb_frame)

      if results.pose_landmarks:
        self.mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
        )

        score, feedback = self.evaluate_frame(
            results.pose_landmarks.landmark, exercise_mode, sensitivity
        )
        scores_list.append(score)
        landmarks_detected = True
      else:
        score = 0.0
        feedback = ["Тело человека не обнаружено в кадре"]
        landmarks_detected = False

      # Отображение текущей оценки на видео
      cv2.putText(
          frame,
          f"Score: {score:.1f}",
          (30, 50),
          cv2.FONT_HERSHEY_SIMPLEX,
          1.0,
          (0, 255, 0) if score > 60 else (0, 0, 255),
          2,
      )

      out.write(frame)

      frame_logs.append({
          "frame": frame_count,
          "timestamp_sec": round(timestamp, 2),
          "exercise": exercise_mode,
          "score": score,
          "normalized_score": round(score / 100.0, 2),
          "detected": landmarks_detected,
          "feedback": feedback,
          "landmarks_detected": landmarks_detected,
      })

    cap.release()
    out.release()

    # Расчет среднего балла по найденным кадрам
    valid_scores = [s for s in scores_list if s > 0]
    avg_score = (
        round(float(np.mean(valid_scores)), 2) if len(valid_scores) > 0 else 0.0
    )

    # Формирование структурированного JSON
    final_output = {
        "summary": {
            "exercise": exercise_mode,
            "total_frames_analyzed": frame_count,
            "average_score": avg_score,
            "sensitivity_used": sensitivity,
        },
        "frames": frame_logs,
    }

    with open(log_path, "w", encoding="utf-8") as f:
      json.dump(final_output, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Обработка завершена. Кадров: {frame_count}, Средний балл: {avg_score}"
    )