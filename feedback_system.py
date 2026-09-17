import threading
from typing import Dict, List, Optional
import pyttsx3


class FeedbackSystem:
    def __init__(self):
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
        except Exception:
            self.tts_engine = None
        self.last_spoken_feedback = ""

    def generate_feedback(self, analysis_results: Dict) -> Dict:
        """
        Генерирует структурированный фидбек, извлекая данные по гибким ключам.
        """
        # 1. Проверка детекции человека
        landmarks_detected = (
            analysis_results.get('detected', False) or 
            analysis_results.get('landmarks_detected', False) or 
            analysis_results.get('landmarks') is not None
        )

        if not landmarks_detected:
            return {
                'messages': ["Человек не найден в кадре"],
                'visual_feedback': "No person detected in frame",
                'color_code': 'red',
                'voice_message': None,
                'form_score': 0.0
            }

        if 'error' in analysis_results:
            return {
                'messages': [f"Ошибка анализа: {analysis_results['error']}"],
                'visual_feedback': f"Analysis error: {analysis_results['error']}",
                'color_code': 'red',
                'voice_message': None,
                'form_score': 0.0
            }

        # 2. Гибкое извлечение оценки и названия упражнения
        exercise = analysis_results.get('exercise', 'squat')
        
        # Получение балла по любому доступному ключу
        form_score = (
            analysis_results.get('score') if analysis_results.get('score') is not None else
            analysis_results.get('form_score') if analysis_results.get('form_score') is not None else
            analysis_results.get('normalized_score_raw')
        )
        if form_score is None:
            form_score = 75.0  # Дефолтное значение при найденных landmarks

        # Извлечение или генерация сообщений
        raw_messages = (
            analysis_results.get('feedback') or 
            analysis_results.get('feedback_messages') or 
            []
        )

        # Если PoseAnalyzer не сгенерировал сообщения сам — формируем их на основе оценки
        messages: List[str] = []
        if isinstance(raw_messages, list) and len(raw_messages) > 0:
            messages = raw_messages
        else:
            if form_score < 60.0:
                messages.append(f"Исправьте технику выполнения ({exercise})")
            elif form_score < 80.0:
                messages.append("Хорошая форма, продолжайте движение")
            else:
                messages.append("Отличная техника!")

        # 3. Цветовой код
        if form_score >= 80:
            color_code = 'green'
        elif form_score >= 60:
            color_code = 'yellow'
        else:
            color_code = 'red'

        # 4. Форматирование visual_feedback
        visual_feedback = f"**{str(exercise).title()}** - Score: {round(form_score, 1)}%\n"
        visual_feedback += "\n".join([f"• {msg}" for msg in messages])

        # 5. Голосовое сообщение (для худшего замечания)
        voice_message = None
        if form_score < 60 and messages:
            voice_message = messages[0]

        return {
            'messages': messages,  # <-- Ключ, необходимый для video_processor.py
            'visual_feedback': visual_feedback,
            'color_code': color_code,
            'voice_message': voice_message,
            'form_score': form_score
        }

    def speak(self, message: Optional[str]):
        """Озвучка сообщений через pyttsx3 (асинхронно)."""
        if not self.tts_engine or not message or message == self.last_spoken_feedback:
            return

        self.last_spoken_feedback = message

        def speak_async():
            try:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            except Exception:
                pass

        thread = threading.Thread(target=speak_async, daemon=True)
        thread.start()

    def get_color_code_rgb(self, color_code: str) -> tuple:
        """Возвращает RGB кортеж по имени цвета."""
        colors = {
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'red': (255, 0, 0),
            'white': (255, 255, 255)
        }
        return colors.get(color_code, (255, 255, 255))