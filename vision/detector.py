# vision/detector.py
"""
YOLOX-Nano ONNX обёртка с правильной постобработкой (grid/stride decode).
"""
import time
import math
import cv2
import numpy as np
import onnxruntime as ort
from .types import Detection


class YoloxDetector:
    PERSON_CLASS_ID = 0  # COCO class 0 = person
    STRIDES = [8, 16, 32]  # сетки 52x52, 26x26, 13x13

    def __init__(self, onnx_path, input_size=(416, 416), conf_threshold=0.3):
        self.input_size = input_size
        self.conf_threshold = conf_threshold

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # Предгенерируем якорную сетку один раз для скорости
        self._grid, self._expanded_strides = self._make_grids(input_size)

        # Прогрев
        dummy = np.zeros((1, 3, *input_size), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})

    def _make_grids(self, input_size):
        """Создает сетку для каждого уровня признаков."""
        grids, expanded_strides = [], []
        h, w = input_size
        for stride in self.STRIDES:
            hsize, wsize = h // stride, w // stride
            yv, xv = np.meshgrid(np.arange(hsize), np.arange(wsize), indexing="ij")
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((1, grid.shape[1], 1), stride))
        return np.concatenate(grids, 1), np.concatenate(expanded_strides, 1)

    def _preprocess(self, bgr):
        """Предобработка: letterbox (важно для сохранения пропорций)."""
        h, w = bgr.shape[:2]
        ih, iw = self.input_size
        ratio = min(ih / h, iw / w)
        new_w, new_h = int(w * ratio), int(h * ratio)

        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((ih, iw, 3), 114, dtype=np.uint8)  # цвет паддинга 114
        padded[:new_h, :new_w] = resized

        x = padded.astype(np.float32).transpose(2, 0, 1)
        return np.ascontiguousarray(x[None]), ratio

    def detect_persons(self, bgr):
        """Возвращает список Detection в координатах исходного кадра."""
        h, w = bgr.shape[:2]
        inp, ratio = self._preprocess(bgr)
        outputs = self.session.run(None, {self.input_name: inp})[0]  # [1, 3549, 85]

        # 1. Декодирование рамок (grid + stride)
        outputs[..., :2] = (outputs[..., :2] + self._grid) * self._expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * self._expanded_strides

        # 2. Фильтрация по уверенности для класса "person"
        boxes = outputs[0]
        scores = boxes[:, 4] * boxes[:, 5 + self.PERSON_CLASS_ID]
        keep = scores > self.conf_threshold
        boxes, scores = boxes[keep], scores[keep]

        if len(boxes) == 0:
            return []

        # 3. Перевод в координаты исходного кадра (с учетом letterbox)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - (self.input_size[1] - w * ratio) / 2) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - (self.input_size[0] - h * ratio) / 2) / ratio

        # 4. NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(
            boxes[:, :4].tolist(), scores.tolist(),
            self.conf_threshold, 0.45  # IoU threshold
        )

        results = []
        for i in indices:
            x1, y1, w_box, h_box = boxes[i, :4]
            x1, y1 = int(x1 - w_box / 2), int(y1 - h_box / 2)
            x2, y2 = int(x1 + w_box), int(y1 + h_box)
            # Клиппинг по границам кадра
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            results.append(Detection(x1, y1, x2, y2, float(scores[i])))
        return results

    def benchmark(self, bgr, n=10):
        self.detect_persons(bgr)
        t0 = time.time()
        for _ in range(n):
            self.detect_persons(bgr)
        return (time.time() - t0) / n