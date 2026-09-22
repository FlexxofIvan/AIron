from Human_repr import load_human_config
from pose_analyzer import PoseAnalyzer


pose_anal = PoseAnalyzer()
human_config = load_human_config("human_config.yaml")


import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


class Pose3DVisualizer:

    def __init__(self, humans, fps=30):

        self.humans = humans
        self.fps = fps

        self.connections = list(
            mp.solutions.pose.POSE_CONNECTIONS
        )

        self.fig = plt.figure(
            figsize=(8, 8)
        )

        self.ax = self.fig.add_subplot(
            111,
            projection="3d"
        )

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.set_title(
            "3D Human Pose"
        )

        # ----------------------------------------------------
        # Skeleton lines
        # ----------------------------------------------------

        self.lines = []

        for _ in self.connections:

            line, = self.ax.plot(
                [],
                [],
                [],
                linewidth=2
            )

            self.lines.append(line)

        # ----------------------------------------------------
        # Joint points
        # ----------------------------------------------------

        self.scatter = self.ax.scatter(
            [],
            [],
            []
        )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        self.ax.view_init(
            elev=15,
            azim=-70
        )

    # ========================================================
    # UPDATE FRAME
    # ========================================================

    def update(self, frame_idx):

        human = self.humans[frame_idx]

        # ----------------------------------------------------
        # Get MediaPipe landmarks
        # ----------------------------------------------------

        points = np.array([
            [
                lm.x,
                lm.y,
                lm.z
            ]

            for lm in human.landmarks.values()
        ])

        # MediaPipe:
        #
        # X -> right
        # Y -> down
        # Z -> depth
        #
        # Для визуализации делаем Y вверх.

        points[:, 1] *= -1

        # ----------------------------------------------------
        # Update joints
        # ----------------------------------------------------

        self.scatter._offsets3d = (
            points[:, 0],
            points[:, 1],
            points[:, 2]
        )

        # ----------------------------------------------------
        # Update skeleton
        # ----------------------------------------------------

        # ВАЖНО:
        # PoseLandmark enum имеет порядок 0..32,
        # а self.landmarks создаётся именно в этом порядке.

        for line, (a, b) in zip(
            self.lines,
            self.connections
        ):

            p1 = points[a]
            p2 = points[b]

            line.set_data(
                [p1[0], p2[0]],
                [p1[1], p2[1]]
            )

            line.set_3d_properties(
                [p1[2], p2[2]]
            )

        # ----------------------------------------------------
        # Fixed camera range
        # ----------------------------------------------------

        self.ax.set_xlim(
            -1,
            1
        )

        self.ax.set_ylim(
            -1,
            1
        )

        self.ax.set_zlim(
            -1,
            1
        )

        return (
            self.lines +
            [self.scatter]
        )

    # ========================================================
    # SHOW
    # ========================================================

    def show(self):

        animation = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.humans),
            interval=1000 / self.fps,
            blit=False
        )

        plt.show()


import cv2

from Human_repr import load_human_config
from pose_analyzer import PoseAnalyzer


video_path = "exp_data/squat.mp4"

video = cv2.VideoCapture(
    video_path
)

fps = video.get(
    cv2.CAP_PROP_FPS
)

humans = []

while True:

    ret, frame = video.read()

    if not ret:
        break

    human = pose_anal.analyze_frame(
        frame,
        human_config
    )

    if human is not None:
        humans.append(human)

video.release()

print(
    f"Processed frames: {len(humans)}"
)

visualizer = Pose3DVisualizer(
    humans,
    fps=fps
)

visualizer.show()