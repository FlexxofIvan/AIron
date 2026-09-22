from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import yaml


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class AngleMeasurement:
    value: float
    confidence: float
    description: str = ""

    @property
    def reliable(self) -> bool:
        return self.confidence >= 0.8

    @property
    def usable(self) -> bool:
        return self.confidence >= 0.6

@dataclass
class OrientationMeasurement:
    value: np.ndarray
    description: str = ""


@dataclass
class JointAngles:
    left_knee: AngleMeasurement
    right_knee: AngleMeasurement

    left_hip: AngleMeasurement
    right_hip: AngleMeasurement

    left_ankle: AngleMeasurement
    right_ankle: AngleMeasurement

    left_elbow: AngleMeasurement
    right_elbow: AngleMeasurement

    left_shoulder: AngleMeasurement
    right_shoulder: AngleMeasurement

    left_wrist: AngleMeasurement
    right_wrist: AngleMeasurement


@dataclass
class BodyAngles:
    trunk_angle_from_vertical: AngleMeasurement
    left_thigh_angle_from_vertical: AngleMeasurement
    right_thigh_angle_from_vertical: AngleMeasurement
    left_shin_angle_from_vertical: AngleMeasurement
    right_shin_angle_from_vertical: AngleMeasurement


@dataclass
class Orientation:
    trunk: np.ndarray

    left_upper_arm: np.ndarray
    right_upper_arm: np.ndarray

    left_forearm: np.ndarray
    right_forearm: np.ndarray

    left_thigh: np.ndarray
    right_thigh: np.ndarray

    left_shin: np.ndarray
    right_shin: np.ndarray


# ============================================================
# CONFIG
# ============================================================

def load_human_config(path="human_config.yaml"):

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# HUMAN
# ============================================================

class Human:

    def __init__(self, landmarks, config):

        self.config = config

        self.landmarks = self._convert_landmarks(
            landmarks
        )

        self.confidence = self._calculate_confidence()

        self.overall_visibility = self._calculate_overall_visibility()

        self.joint_angles = self._calculate_joint_angles()

        self.body_angles = self._calculate_body_angles()

        self.orientation = self._calculate_orientation()

    # ========================================================
    # LANDMARKS
    # ========================================================

    def _convert_landmarks(self, landmarks):

        mp_pose = mp.solutions.pose

        return {
            name.name: landmarks.landmark[idx]
            for idx, name in enumerate(mp_pose.PoseLandmark)
        }

    def point(self, name):

        if name in self.landmarks:
            lm = self.landmarks[name]

            return np.array(
                [lm.x, lm.y, lm.z],
                dtype=np.float32
            )

        derived_points = self.config.get(
            "derived_points",
            {}
        )

        if name in derived_points:

            definition = derived_points[name]

            if definition["type"] == "midpoint":
                points = definition["points"]

                a = self.point(points[0])
                b = self.point(points[1])

                return self.midpoint(a, b)

            raise ValueError(
                f"Unknown derived point type: "
                f"{definition['type']}"
            )

        raise KeyError(
            f"Unknown point: {name}"
        )

    def point2d(self, name):

        return self.point(name)[:2]

    def visibility(self, name):
        """
        Return visibility/confidence of a point.

        For derived points, confidence is the minimum
        visibility of all source landmarks.
        """

        if name in self.landmarks:

            return float(
                self.landmarks[name].visibility
            )

        derived_points = self.config.get(
            "derived_points",
            {}
        )

        if name in derived_points:

            definition = derived_points[name]

            return min(
                self.visibility(point_name)
                for point_name in definition["points"]
            )

        raise KeyError(
            f"Unknown point: {name}"
        )

    # ========================================================
    # GEOMETRY
    # ========================================================

    @staticmethod
    def angle(a, b, c):
        """
        Geometric angle ABC in degrees.
        """

        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-8 or norm_bc < 1e-8:
            return np.nan

        cosine = np.dot(ba, bc) / (
            norm_ba * norm_bc
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        return float(
            np.degrees(
                np.arccos(cosine)
            )
        )

    @staticmethod
    def vector(a, b):

        return b - a

    @staticmethod
    def normalize(v):

        norm = np.linalg.norm(v)

        if norm < 1e-8:
            return np.zeros_like(v)

        return v / norm

    @staticmethod
    def midpoint(a, b):

        return (a + b) / 2.0

    # ========================================================
    # MEASUREMENTS
    # ========================================================

    def angle_measurement(
            self,
            a_name,
            b_name,
            c_name,
            description=""
    ):
        """
        Geometric angle A-B-C.
        """

        value = self.angle(
            self.point(a_name),
            self.point(b_name),
            self.point(c_name)
        )

        confidence = min(
            self.visibility(a_name),
            self.visibility(b_name),
            self.visibility(c_name)
        )

        return AngleMeasurement(
            value=value,
            confidence=confidence,
            description=description
        )

    def segment_angle_to_vertical(
            self,
            start_name,
            end_name,
            description=""
    ):
        start = self.point2d(start_name)
        end = self.point2d(end_name)

        v = end - start
        norm = np.linalg.norm(v)

        if norm < 1e-8:
            value = np.nan
        else:
            angle = np.degrees(
                np.arctan2(
                    v[0],
                    -v[1]
                )
            )

            value = abs(angle)

            if value > 90:
                value = 180 - value

        confidence = min(
            self.visibility(start_name),
            self.visibility(end_name)
        )

        return AngleMeasurement(
            value=float(value),
            confidence=confidence,
            description=description
        )

    def trunk_angle(self):

        return self.segment_angle_to_vertical(
            "MID_HIP",
            "MID_SHOULDER"
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def _calculate_confidence(self):

        return {
            name: self.visibility(name)
            for name in self.landmarks
        }

    # ========================================================
    # JOINT ANGLES
    # ========================================================

    def _calculate_joint_angles(self):

        measurements = {}

        definitions = self.config.get(
            "joint_angles",
            {}
        )

        for name, definition in definitions.items():

            points = definition["points"]

            if len(points) != 3:

                raise ValueError(
                    f"Joint angle '{name}' must have "
                    f"exactly 3 points"
                )

            measurements[name] = self.angle_measurement(
                points[0],
                points[1],
                points[2],
                definition.get("description", "")
            )

        return JointAngles(**measurements)

    # ========================================================
    # BODY ANGLES
    # ========================================================

    def _calculate_body_angles(self):

        measurements = {}

        definitions = self.config.get(
            "body_angles",
            {}
        )

        for name, definition in definitions.items():

            measurement_type = definition["type"]
            points = definition["points"]
            description = definition.get("description", "")

            if measurement_type == "segment_to_vertical":

                measurements[name] = (
                    self.segment_angle_to_vertical(
                        points[0],
                        points[1],
                        description
                    )
                )

            elif measurement_type == "trunk":

                measurements[name] = (
                    self.trunk_angle(
                        description
                    )
                )

            else:

                raise ValueError(
                    f"Unknown body angle type: "
                    f"{measurement_type}"
                )

        return BodyAngles(**measurements)

    # ========================================================
    # ORIENTATION
    # ========================================================

    def _calculate_orientation(self):

        measurements = {}

        definitions = self.config.get(
            "orientation",
            {}
        )

        for name, definition in definitions.items():

            points = definition["points"]

            if len(points) != 2:
                raise ValueError(
                    f"Orientation '{name}' must have "
                    f"exactly 2 points"
                )

            start = self.point(points[0])
            end = self.point(points[1])

            vector = self.normalize(
                self.vector(start, end)
            )

            description = definition.get(
                "description",
                ""
            )

            measurements[name] = OrientationMeasurement(
                value=vector,
                description=description
            )

        return Orientation(**measurements)

    def _calculate_overall_visibility(self):
        """
        Calculate overall human visibility using
        the geometric mean of landmark visibilities.
        """

        visibilities = np.array(
            list(self.confidence.values()),
            dtype=np.float32
        )

        if len(visibilities) == 0:
            return 0.0

        visibilities = np.clip(
            visibilities,
            1e-8,
            1.0
        )

        return float(
            np.exp(
                np.mean(
                    np.log(visibilities)
                )
            )
        )

    def __repr__(self):

        output = []

        output.append("\nJOINT ANGLES")

        for name, measurement in vars(self.joint_angles).items():
            output.append(
                f"{name:}:"
                f"{measurement.value:8.2f}° | "
                f"confidence: {measurement.confidence:.3f}"
            )
            output.append(
                f"    description: {measurement.description}"
            )

        output.append("\nBODY ANGLES")

        for name, measurement in vars(self.body_angles).items():
            output.append(
                f"{name:}:"
                f"{measurement.value:8.2f}° | "
                f"confidence: {measurement.confidence:.3f}"
            )
            output.append(
                f"    description: {measurement.description}"
            )

        output.append("\nORIENTATION")

        for name, measurement in vars(self.orientation).items():
            vector = measurement.value

            output.append(
                f"{name}:"
                f"[{vector[0]: .3f}, "
                f"{vector[1]: .3f}, "
                f"{vector[2]: .3f}]"
            )

            output.append(
                f"    description: {measurement.description}"
            )

        output.append("\nOVERALL VISIBILITY")

        output.append(
            f"{self.overall_visibility:.4f}"
        )

        return "\n".join(output)

    def to_dict(self):

        return {
            "joint_angles": {
                name: {
                    "value": measurement.value,
                    "unit": "degrees",
                    "confidence": measurement.confidence,
                    "description": measurement.description,
                }
                for name, measurement
                in vars(self.joint_angles).items()
            },

            "body_angles": {
                name: {
                    "value": measurement.value,
                    "unit": "degrees",
                    "confidence": measurement.confidence,
                    "description": measurement.description,
                }
                for name, measurement
                in vars(self.body_angles).items()
            },

            "orientation": {
                name: {
                    "vector": measurement.value.tolist(),
                    "description": measurement.description,
                }
                for name, measurement
                in vars(self.orientation).items()
            },

            "overall_visibility": {
                "value": self.overall_visibility,
                "description": (
                    "Geometric mean of MediaPipe landmark "
                    "visibility scores across body landmarks."
                ),
            },
        }

    def to_dict_compact(self):
        return {
            "joint_angles": {
                name: {
                    "value": int(round(measurement.value)),
                    "confidence": round(measurement.confidence, 2),
                }
                for name, measurement
                in vars(self.joint_angles).items()
            },

            "body_angles": {
                name: {
                    "value": int(round(measurement.value)),
                    "confidence": round(measurement.confidence, 2),
                }
                for name, measurement
                in vars(self.body_angles).items()
            },

            "orientation": {
                name: {
                    "vector": [
                        round(float(x), 2)
                        for x in measurement.value
                    ]
                }
                for name, measurement
                in vars(self.orientation).items()
            },

            "overall_visibility": round(
                self.overall_visibility,
                2
            ),
        }

    def get_descriptions(self):
        return {
            "joint_angles": {
                name: measurement.description.strip()
                for name, measurement
                in vars(self.joint_angles).items()
            },

            "body_angles": {
                name: measurement.description.strip()
                for name, measurement
                in vars(self.body_angles).items()
            },

            "orientation": {
                name: measurement.description.strip()
                for name, measurement
                in vars(self.orientation).items()
            },
        }