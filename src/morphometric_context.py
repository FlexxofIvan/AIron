"""Individual Morphometric Context Module.

Extracts user-specific anthropometric measurements from pre-computed
Virtual Measurements (Choutas et al., CVPR 2022) and formats them as
human-readable descriptors that ground coaching in individual body geometry.
"""

import json
import os
import re
from typing import Dict, Optional



_TRAIN_MEASUREMENT_FILENAME = "all_hsmr_shapy_measurements.json"
_TEST_MEASUREMENT_FILENAME = "all_hsmr_shapy_measurements_benchmark.json"


def _load_measurement_lookup(
    json_path: str,
) -> Dict[str, Dict[str, float]]:
    """Load pre-computed measurements from a SHAPY JSON file.

    Returns:
        {person_id: {height, mass, chest, waist, hips}}
    """
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        data = json.load(f)
    lookup = {}
    for entry in data.get("measurements", []):
        pid = entry["person_id"]
        lookup[pid] = entry["measurements"]
    return lookup


def format_morphometric_context(
    measurements: Dict[str, float],
) -> str:
    """Format measurements as human-readable morphometric context C_morph.
        "User body: height 1.78 m, mass 73.22 kg, chest 1.00 m,
         waist 0.83 m, hips 0.98 m."
    """
    return (
        f"User body: "
        f"height {measurements['height']:.2f} m, "
        f"mass {measurements['mass']:.2f} kg, "
        f"chest {measurements['chest']:.2f} m, "
        f"waist {measurements['waist']:.2f} m, "
        f"hips {measurements['hips']:.2f} m."
    )


def _extract_hsmr_id(path: str) -> Optional[str]:
    """Extract HSMR person id from a file path.

    Examples:
        '.../HSMR-0006.npy' → 'HSMR-0006'
        '.../HSMR_outputs/HSMR-0123.npy' → 'HSMR-0123'
    """
    basename = os.path.basename(path)
    m = re.match(r"(HSMR-\d+)", basename)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Morphometric Context Module
# ---------------------------------------------------------------------------
class MorphometricContextModule:
    """Individual Morphometric Context Module (Sec 3.4.1).

    Loads pre-computed Virtual Measurements from SHAPY JSON files and
    produces formatted C_morph context strings for the language model.
    """

    def __init__(
        self,
        morphometric_dir: Optional[str] = None,
        train_json: Optional[str] = None,
        test_json: Optional[str] = None,
    ):
        """Construct the morphometric context module.

        Args:
            morphometric_dir: Directory containing the SHAPY JSON files
                (set via the ``morphometric_dir`` yaml key).
            train_json / test_json: Explicit override paths. If unset they
                are derived from ``morphometric_dir`` + the conventional
                filenames.
        """
        self._lookup: Dict[str, Dict[str, float]] = {}
        self._context_cache: Dict[str, str] = {}

        paths = [train_json, test_json]
        if morphometric_dir:
            base = morphometric_dir if morphometric_dir.endswith("/") else morphometric_dir + "/"
            paths = [
                train_json or (base + _TRAIN_MEASUREMENT_FILENAME),
                test_json or (base + _TEST_MEASUREMENT_FILENAME),
            ]
        for path in paths:
            if path:
                self._lookup.update(_load_measurement_lookup(path))

    def get_context(
        self,
        skeleton_path: str,
        cache_key: Optional[str] = None,
    ) -> str:
        """Get morphometric context C_morph for a video.

        Args:
            skeleton_path: Path to the HSMR .npy file.
            cache_key: Optional key for caching (default: skeleton_path).

        Returns:
            Formatted C_morph string, or "" if measurements unavailable.
        """
        key = cache_key or skeleton_path
        if key in self._context_cache:
            return self._context_cache[key]

        measurements = self.get_measurements(skeleton_path)
        if measurements is None:
            return ""

        context = format_morphometric_context(measurements)
        self._context_cache[key] = context
        return context

    def get_measurements(
        self,
        skeleton_path: str,
    ) -> Optional[Dict[str, float]]:
        """Get raw measurement values for a subject from the SHAPY lookup.

        Returns None if the subject's HSMR id is not present in the
        pre-computed measurement files.
        """
        hsmr_id = _extract_hsmr_id(skeleton_path)
        if hsmr_id and hsmr_id in self._lookup:
            return self._lookup[hsmr_id]
        return None
