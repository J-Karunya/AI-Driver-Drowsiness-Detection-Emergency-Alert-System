import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import torch
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av

# Our modules
from utils import extract_mediapipe_features
from drowsiness_lstm import get_model
from ultralytics import YOLO

st.set_page_config(page_title="Real-Time ADAS", layout="wide")

st.title("🚗 Real-Time Driver Drowsiness & Distraction Detection")
st.markdown("Advanced Driver Assistance System (ADAS) using **MediaPipe**, **YOLOv8**, and **PyTorch LSTM**.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Settings")
EAR_THRESH = st.sidebar.slider("EAR Threshold (Eye Closure)", 0.1, 0.4, 0.2, 0.01)
MAR_THRESH = st.sidebar.slider("MAR Threshold (Yawning)", 0.4, 1.0, 0.6, 0.05)
YAW_THRESH = st.sidebar.slider("Yaw Threshold (Looking Away L/R)", 15, 50, 25, 1)
PITCH_THRESH = st.sidebar.slider("Pitch Threshold (Looking Down)", 15, 40, 20, 1)
DROWSY_FRAMES = st.sidebar.slider("Frames to trigger Drowsy", 10, 60, 20, 1)

# Initialize models
@st.cache_resource
def load_models():
    # Load YOLOv8 for phone detection (nano is fastest)
    yolo_model = YOLO("yolov8n.pt")
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load PyTorch LSTM
    # We use untrained weights here as a structural demo if model_path is not found
    lstm_model = get_model('drowsiness_lstm.pt', device=device) 
    
    return yolo_model, lstm_model

yolo_model, lstm_model = load_models()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ADASVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_counter = 0
        self.drowsy_counter = 0
        self.sequence_buffer = deque(maxlen=30)
        self.yolo_model = yolo_model
        self.lstm_model = lstm_model
        self.face_mesh = face_mesh
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # 1. Distraction - Phone Detection (YOLOv8)
        # We only want to process every N frames or resize for speed, but let's try direct first
        results = self.yolo_model(img, classes=[67], verbose=False) # 67 is 'cell phone' in COCO
        phone_detected = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "Phone Distraction!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                phone_detected = True

        # 2. Drowsiness & Head Pose (MediaPipe)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fm_result = self.face_mesh.process(img_rgb)
        
        drowsy_alert = False
        lstm_drowsy = False
        looking_away = False
        ear, mar = 0, 0
        pitch, yaw, roll = 0, 0, 0
        
        if fm_result.multi_face_landmarks:
            landmarks = fm_result.multi_face_landmarks[0]
            ear, mar, pitch, yaw, roll = extract_mediapipe_features(landmarks, w, h)
            
            # --- Head Pose (Looking Away) ---
            if abs(yaw) > YAW_THRESH or pitch > PITCH_THRESH or pitch < -PITCH_THRESH:
                looking_away = True
                cv2.putText(img, "DISTRACTION: Looking Away", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            # --- Rule-Based Drowsiness ---
            if ear < EAR_THRESH or mar > MAR_THRESH:
                self.drowsy_counter += 1
                if self.drowsy_counter >= DROWSY_FRAMES:
                    drowsy_alert = True
            else:
                self.drowsy_counter = max(0, self.drowsy_counter - 1)
                
            # --- LSTM-Based Drowsiness ---
            self.sequence_buffer.append([ear, mar])
            if len(self.sequence_buffer) == 30:
                # Prepare tensor: shape (1, 30, 2)
                device = next(self.lstm_model.parameters()).device
                seq_tensor = torch.tensor(list(self.sequence_buffer), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = self.lstm_model(seq_tensor)
                    # output shape (1, 2). Softmax for probabilities
                    prob = torch.softmax(output, dim=1)[0]
                    if prob[1] > 0.6: # Class 1 is drowsy
                        lstm_drowsy = True

            # Draw metrics
            cv2.putText(img, f"EAR: {ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"MAR: {mar:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"YAW: {yaw:.1f} PITCH: {pitch:.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if drowsy_alert:
            cv2.putText(img, "WARNING: RULE-BASED DROWSINESS", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
        if lstm_drowsy:
            cv2.putText(img, "WARNING: LSTM DROWSINESS DETECTED", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
        if phone_detected:
            cv2.putText(img, "WARNING: PHONE USAGE", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.subheader("Live ADAS Feed")
webrtc_streamer(
    key="adas-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=ADASVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("""
### How it works
1. **Drowsiness**: We compute Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) using MediaPipe 468-point face mesh. 
    * A **Rule-Based System** triggers if values exceed thresholds for a set number of frames.
    * A **PyTorch LSTM** takes a sequence of EAR/MAR over 30 frames to detect gradual onset.
2. **Distraction**:
    * **Looking Away**: Head Pose Estimation (Yaw/Pitch) computed via Perspective-n-Point (PnP) using 2D-3D face points matching.
    * **Phone Usage**: A pre-trained YOLOv8 Nano model detects cell phones in the frame.
""")
