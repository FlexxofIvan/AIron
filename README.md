# Fitness Form Checker

Real-time fitness form analysis using computer vision and pose estimation.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
streamlit run main.py
```

## Features

- **Real-time pose analysis** for squats, push-ups, and yoga poses
- **Automatic exercise detection** or manual selection
- **Form scoring** with color-coded feedback (green/yellow/red)
- **Voice feedback** for corrections
- **Rep counting** with form validation
- **Session tracking** and performance history

## Supported Exercises

1. **Squats**: Analyzes knee/hip angles, squat depth, knee alignment
2. **Push-ups**: Checks elbow angles, body alignment, range of motion  
3. **Downward Dog**: Validates triangle formation, arm/leg straightness
4. **Warrior Pose**: Monitors lunge depth, arm positioning, torso alignment

## Controls

- Exercise mode selection (auto-detect or manual)
- Sensitivity adjustment (0.5-1.0)
- Voice feedback toggle
- Real-time session statistics

## Technical Details

- Uses MediaPipe Pose for landmark detection
- OpenCV for video processing at 30 FPS
- Streamlit for web-based GUI
- Text-to-speech feedback system