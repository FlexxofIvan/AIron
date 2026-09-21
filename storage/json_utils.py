# storage/json_utils.py
"""
Приведение numpy-типов и tuple к JSON-совместимым.
"""
import numpy as np


def to_jsonable(obj):
    """Рекурсивно превращает numpy-скаляры и tuple в JSON-совместимые типы."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj