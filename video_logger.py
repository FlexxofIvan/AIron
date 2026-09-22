import json

import cv2

from pose_analyzer import PoseAnalyzer
from Human_repr import load_human_config


def log_video(
    video_path: str,
    output_path: str,
    sample_every: int = 5,
):
    """
    Process video and save pose/biomechanical data to JSON.

    Parameters
    ----------
    video_path:
        Path to input video.

    output_path:
        Path to output JSON.

    sample_every:
        Analyze every N-th frame.
        For example:
            1 -> every frame
            5 -> every 5th frame
            10 -> every 10th frame
    """

    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")

    pose_analyzer = PoseAnalyzer()
    human_config = load_human_config("human_config.yaml")

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError(
            f"Could not determine FPS for video: {video_path}"
        )

    total_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    result = {
        "video": video_path,
        "fps": round(fps, 2),
        "sample_every": sample_every,
        "descriptions": None,
        "frames": [],
    }

    frame_index = 0

    try:
        while True:

            success, frame = video.read()

            if not success:
                break

            # Skip frames that are not sampled
            if frame_index % sample_every != 0:
                frame_index += 1
                continue

            human = pose_analyzer.analyze_frame(
                frame,
                human_config,
            )

            # Если поза на кадре не найдена
            if human is None:
                frame_index += 1
                continue

            # Описания записываем только один раз
            if result["descriptions"] is None:
                result["descriptions"] = (
                    human.get_descriptions()
                )

            frame_data = human.to_dict_compact()

            frame_data["frame"] = frame_index
            frame_data["second"] = round(
                frame_index / fps,
                2
            )

            # Чтобы frame/second были в начале объекта
            frame_data = {
                "frame": frame_data.pop("frame"),
                "second": frame_data.pop("second"),
                **frame_data,
            }

            result["frames"].append(frame_data)

            frame_index += 1

    finally:
        video.release()

    # Если ни одного кадра не обработалось
    if result["descriptions"] is None:
        result["descriptions"] = {
            "joint_angles": {},
            "body_angles": {},
            "orientation": {},
        }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Processed {len(result['frames'])} frames "
        f"out of {total_frames}"
    )

    print(
        f"Saved to: {output_path}"
    )

log_video(
        video_path="exp_data/l.mp4",
        output_path="exp_data/l_pose_logs.json",
        sample_every=5,
        )