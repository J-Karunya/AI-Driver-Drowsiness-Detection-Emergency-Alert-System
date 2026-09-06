# AI-Driver-Drowsiness-Detection-Emergency-Alert-System

A real-time AI-based driver safety system that detects **drowsiness, distraction, and mobile-phone usage** and progressively escalates alerts when the driver fails to recover.

# Features
Real-time drowsiness detection using MediaPipe + LSTM
Mobile-phone detection using YOLO
Driver distraction detection using head-pose estimation
Continuous alarm and voice warnings
Progressive emergency escalation
Live GPS location tracking
Emergency SMS notification with location
Prevents duplicate emergency alerts

# Workflow
Camera
  ↓
Face & Phone Detection
  ↓
Drowsiness + Distraction Analysis
  ↓
Driver State Machine
  ↓
NORMAL → ALARM → CRITICAL → EMERGENCY
  ↓
GPS + SMS Alert


# Tech Stack
Python, OpenCV, MediaPipe, PyTorch, LSTM, YOLO, Streamlit, WebRTC, Twilio


