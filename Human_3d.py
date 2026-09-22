import cv2
import json
from Human_repr import load_human_config
from pose_analyzer import PoseAnalyzer


pose_anal = PoseAnalyzer()
human_config = load_human_config("human_config.yaml")

video_path = "exp_data/squat.mp4"
output_path = "exp_data/squat.json"

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise RuntimeError("Could not determine video FPS")


samples_per_second = 5

sample_interval = fps / samples_per_second

next_sample_frame = 0.0
frame_idx = 0

logs = {}


def format_timestamp(seconds):
    minutes = int(seconds // 60)
    seconds_remainder = seconds % 60

    return f"{minutes:02d}:{seconds_remainder:06.3f}"


while True:

    ret, frame = video.read()

    if not ret:
        break

    if frame_idx >= next_sample_frame:

        human = pose_anal.analyze_frame(
            frame,
            human_config
        )

        next_sample_frame += sample_interval

        if human is not None:

            data = human.to_dict()

            second = int(frame_idx / fps)

            if second not in logs:
                logs[second] = {
                    "second": second,
                    "frames": []
                }

            timestamp = frame_idx / fps

            data["frame"] = frame_idx
            data["timestamp"] = timestamp
            data["time"] = format_timestamp(timestamp)

            logs[second]["frames"].append(data)

        else:
            print(f"⚠️ Person not detected at frame {frame_idx}")

    frame_idx += 1

video.release()

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        list(logs.values()),
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"Saved pose logs to: {output_path}")