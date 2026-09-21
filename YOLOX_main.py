
import cv2
import numpy as np
import onnxruntime as ort

class YoloxNanoDetector:
    def __init__(self, model_path="yolox_nano.onnx"):
        # 1. Загружаем модель в память через полностью бесплатный ONNX Runtime
        # Он работает невероятно быстро даже на обычном процессоре (CPU) ноутбука
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (416, 416) # Стандартный размер входа для YOLOX-Nano
        
        # В датасете COCO класс 0 — это "person" (человек)
        self.TARGET_CLASS_ID = 0 

    def _preprocess(self, img):
        """Предобработка кадра под требования YOLOX-Nano"""
        # Меняем размер под входной размер нейросети (416х416)
        resized_img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        # YOLOX-Nano ожидает формат Float32 и каналы CHW (Channel, Height, Width)
        img_data = resized_img.astype(np.float32)
        img_data = img_data.transpose(2, 0, 1) # Превращаем HWC в CHW
        img_data = np.expand_dims(img_data, axis=0) # Добавляем размер батча [1, 3, 416, 416]
        return img_data

    def detect_person(self, frame, conf_threshold=0.3):
        """
        Ищет человека на кадре. 
        Возвращает: (x_center, frame_width) или (None, None) если человек не найден
        """
        h_orig, w_orig, _ = frame.shape
        img_data = self._preprocess(frame)
        
        # 2. Запуск прямого прохода нейросети (Inference)
        outputs = self.session.run(None, {self.input_name: img_data})
        predictions = outputs[0][0] # Получаем массив предсказаний
        
        best_person_box = None
        max_score = 0
        
        # 3. Разбор выходных данных (Матрица предсказаний YOLOX)
        # Каждая строка — это [x_center, y_center, width, height, obj_conf, class_score_0, class_score_1...]
        for pred in predictions:
            obj_conf = pred[4]
            class_conf = pred[5 + self.TARGET_CLASS_ID]
            score = obj_conf * class_conf # Итоговая уверенность, что это человек
            
            if score > conf_threshold and score > max_score:
                max_score = score
                # Декодируем координаты (переводим из формата 416x416 обратно в разрешение камеры)
                x_center_raw = pred[0] * (w_orig / self.input_shape[1])
                y_center_raw = pred[1] * (h_orig / self.input_shape[0])
                width_raw = pred[2] * (w_orig / self.input_shape[1])
                height_raw = pred[3] * (h_orig / self.input_shape[0])
                
                # Координаты Bounding Box рамки
                xmin = int(x_center_raw - width_raw / 2)
                ymin = int(y_center_raw - height_raw / 2)
                xmax = int(x_center_raw + width_raw / 2)
                ymax = int(y_center_raw + height_raw / 2)
                
                best_person_box = (xmin, ymin, xmax, ymax)

        # 4. Если человек найден, вычисляем его центр для наведения башни
        if best_person_box:
            xmin, ymin, xmax, ymax = best_person_box
            person_x_center = int((xmin + xmax) / 2)
            
            # Дополнительно: рисуем зеленую рамку для визуального контроля на ПК
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.circle(frame, (person_x_center, int((ymin+ymax)/2)), 5, (0, 0, 255), -1)
            
            return person_x_center, w_orig
            
        return None, None

# Пример интеграции в цикл сканирования робота
if __name__ == "__main__":
    detector = YoloxNanoDetector("yolox_nano.onnx")
    
    # Симулируем получение кадра с камеры робота (например, разрешение 640x480)
    fake_camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Нарисуем белый круг, имитирующий человека по центру
    cv2.circle(fake_camera_frame, (400, 240), 50, (255, 255, 255), -1) 
    
    x_center, frame_width = detector.detect_person(fake_camera_frame)
    
    if x_center:
        print(f"[УСПЕХ] Человек найден! Его центр по оси X: {x_center} пикселей (Ширина кадра: {frame_width})")
        # Логика смещения: центр кадра — 320
        if x_center < 300:
            print(" -> Команда роботу: доверни башню ЛЕВЕЕ")
        elif x_center > 340:
            print(" -> Команда роботу: доверни башню ПРАВЕЕ")
        else:
            print(" -> Робот наведен идеально! Останавливаемся и включаем анализ упражнений.")