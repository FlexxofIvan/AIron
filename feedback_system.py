import pyttsx3
import threading
from typing import Dict, List

class FeedbackSystem:
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.8)
        self.last_spoken_feedback = ""
        
    def generate_feedback(self, analysis_results: Dict) -> Dict:
        """Generate comprehensive feedback from analysis results"""
        if not analysis_results.get('detected', False):
            return {
                'visual_feedback': "No person detected in frame",
                'color_code': 'red',
                'voice_message': None
            }
        
        if 'error' in analysis_results:
            return {
                'visual_feedback': f"Analysis error: {analysis_results['error']}",
                'color_code': 'red',
                'voice_message': None
            }
        
        exercise = analysis_results.get('exercise', 'unknown')
        form_score = analysis_results.get('form_score', 0)
        feedback_messages = analysis_results.get('feedback', [])
        
        # Determine color code based on form score
        if form_score >= 80:
            color_code = 'green'
        elif form_score >= 60:
            color_code = 'yellow'
        else:
            color_code = 'red'
        
        # Format visual feedback
        visual_feedback = f"**{exercise.title()}** - Score: {form_score}%\n"
        visual_feedback += "\n".join([f"• {msg}" for msg in feedback_messages])
        
        # Generate voice message (only for significant corrections)
        voice_message = None
        if form_score < 60 and feedback_messages:
            voice_message = feedback_messages[0]  # Speak most important feedback
        
        return {
            'visual_feedback': visual_feedback,
            'color_code': color_code,
            'voice_message': voice_message,
            'form_score': form_score
        }
    
    def speak(self, message: str):
        """Speak feedback message using text-to-speech"""
        if message and message != self.last_spoken_feedback:
            self.last_spoken_feedback = message
            
            def speak_async():
                try:
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
                except Exception:
                    pass  # Silently handle TTS errors
            
            thread = threading.Thread(target=speak_async)
            thread.daemon = True
            thread.start()
    
    def get_color_code_rgb(self, color_code: str) -> tuple:
        """Get RGB values for color codes"""
        colors = {
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'red': (255, 0, 0),
            'white': (255, 255, 255)
        }
        return colors.get(color_code, (255, 255, 255))