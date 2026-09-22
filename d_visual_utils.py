import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

from Human_repr import Human, load_human_config
from pose_analyzer import PoseAnalyzer

from Human_repr import Human, load_human_config
from pose_analyzer import PoseAnalyzer


def print_measurement(name, measurement):

    print(f"RAW VALUE: {measurement.value!r}")

    status = (
        "RELIABLE"
        if measurement.reliable
        else "LOW CONFIDENCE"
    )

    print(
        f"{name:25s}: "
        f"{measurement.value:8.2f}° | "
        f"confidence: {measurement.confidence:.3f} | "
        f"{status}"
    )


def plot_3d_skeleton(results, frame_idx=None):

    landmarks = results.pose_world_landmarks.landmark

    mp_pose = mp.solutions.pose

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    x = np.array([lm.x for lm in landmarks])
    y = np.array([lm.y for lm in landmarks])
    z = np.array([lm.z for lm in landmarks])

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------

    ax.scatter(
        x,
        y,
        z,
        s=30
    )

    # --------------------------------------------------------
    # Skeleton connections
    # --------------------------------------------------------

    for connection in mp_pose.POSE_CONNECTIONS:

        start_idx, end_idx = connection

        ax.plot(
            [x[start_idx], x[end_idx]],
            [y[start_idx], y[end_idx]],
            [z[start_idx], z[end_idx]]
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    for idx, lm in enumerate(landmarks):

        name = mp_pose.PoseLandmark(idx).name

        ax.text(
            lm.x,
            lm.y,
            lm.z,
            name,
            fontsize=7
        )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    title = "MediaPipe 3D Pose"

    if frame_idx is not None:
        title += f" — Frame {frame_idx}"

    ax.set_title(title)

    # Equal-ish aspect ratio
    ax.set_box_aspect([
        np.ptp(x),
        np.ptp(y),
        np.ptp(z)
    ])

    plt.tight_layout()

    return fig


def process_frame(
    frame,
    pose_anal,
    human_config,
    frame_idx=None,
    save_skeleton_path=None
):
    """
    Process one BGR video frame.

    Visualization/output behavior is controlled
    by human_config["visualization"].
    """

    visualization_config = human_config.get(
        "visualization",
        {}
    )

    print_config = visualization_config.get(
        "print",
        {}
    )

    # ========================================================
    # BGR -> RGB
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    results = pose_anal.pose.process(rgb)

    print(
        "Pose detected:",
        results.pose_landmarks is not None
    )

    if results.pose_landmarks is None:
        print("Person not detected")
        return frame, results, None, None

    # ========================================================
    # DRAW 2D SKELETON
    # ========================================================

    pose_anal.mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        pose_anal.mp_pose.POSE_CONNECTIONS
    )

    if save_skeleton_path is not None:

        cv2.imwrite(
            save_skeleton_path,
            frame
        )

        print(
            f"Skeleton saved to: "
            f"{save_skeleton_path}"
        )

    # ========================================================
    # HUMAN
    # ========================================================

    human = Human(
        results.pose_world_landmarks,
        human_config
    )

    print("\n" + "=" * 80)

    if frame_idx is not None:
        print(f"FRAME {frame_idx}")

    print("=" * 80)

    # ========================================================
    # TEST ANGLE
    # ========================================================

    test_angle_config = visualization_config.get(
        "test_angle",
        {}
    )

    if test_angle_config.get("enabled", False):

        points = test_angle_config["points"]

        a = human.point(points[0])
        b = human.point(points[1])
        c = human.point(points[2])

        print(
            f"\nTEST ANGLE: "
            f"{points[0]} - {points[1]} - {points[2]}"
        )

        print(
            "3D:",
            human.angle(a, b, c)
        )

        print(
            "2D:",
            human.angle(a[:2], b[:2], c[:2])
        )

    # ========================================================
    # JOINT ANGLES
    # ========================================================

    if print_config.get("joint_angles", False):

        print("\nJOINT ANGLES")
        print("-" * 80)

        for name, measurement in vars(
            human.joint_angles
        ).items():

            print_measurement(
                name,
                measurement
            )

    # ========================================================
    # BODY ANGLES
    # ========================================================

    if print_config.get("body_angles", False):

        print("\nBODY ANGLES")
        print("-" * 80)

        for name, measurement in vars(
            human.body_angles
        ).items():

            print_measurement(
                name,
                measurement
            )

    # ========================================================
    # ORIENTATION
    # ========================================================

    if print_config.get("orientation", False):

        print("\nORIENTATION")
        print("-" * 80)

        for name, vector in vars(
            human.orientation
        ).items():

            print(
                f"{name:25s}: "
                f"{vector}"
            )

    # ========================================================
    # LANDMARK VISIBILITY
    # ========================================================

    if print_config.get(
        "landmark_visibility",
        False
    ):

        print("\nLANDMARK VISIBILITY")
        print("-" * 80)

        important_landmarks = visualization_config.get(
            "important_landmarks",
            []
        )

        for name in important_landmarks:

            print(
                f"{name:25s}: "
                f"{human.visibility(name):.3f}"
            )

    print("\n" + "=" * 80)

    # ========================================================
    # 3D SKELETON
    # ========================================================

    fig = None

    skeleton_config = visualization_config.get(
        "skeleton_3d",
        {}
    )

    if skeleton_config.get("enabled", False):

        fig = plot_3d_skeleton(
            results,
            frame_idx=frame_idx
        )

    return frame, results, human, fig


def find_best_visibility_frame(
    video_path,
    pose_anal,
    human_config
):
    video = cv2.VideoCapture(video_path)

    best_visibility = -1.0
    best_frame = None
    best_results = None
    best_frame_idx = None

    frame_idx = 0

    while True:

        ret, frame = video.read()

        if not ret:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose_anal.pose.process(rgb)

        if results.pose_world_landmarks is None:
            frame_idx += 1
            continue

        human = Human(
            results.pose_world_landmarks,
            human_config
        )

        visibility = human.overall_visibility

        if visibility > best_visibility:

            best_visibility = visibility
            best_frame = frame.copy()
            best_results = results
            best_frame_idx = frame_idx

        frame_idx += 1

    video.release()

    return (
        best_frame,
        best_results,
        best_visibility,
        best_frame_idx
    )