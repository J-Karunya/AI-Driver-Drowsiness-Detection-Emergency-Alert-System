import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import torch

from collections import deque
from datetime import datetime

from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    WebRtcMode,
    RTCConfiguration
)

import av

from utils import extract_mediapipe_features
from drowsiness_lstm import get_model
from ultralytics import YOLO

from driver_state import DriverStateMachine, DriverState
from alarm_manager import AlarmManager
from emergency_manager import EmergencyManager

from streamlit_js_eval import streamlit_js_eval


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Real-Time ADAS",
    layout="wide"
)

st.title("🚗 Real-Time Driver Drowsiness & Distraction Detection")

st.markdown(
    """
    Advanced Driver Assistance System (ADAS) using
    **MediaPipe**, **YOLOv8**, and **PyTorch BiLSTM**.
    """
)


# =========================================================
# GLOBAL GPS STORAGE
# =========================================================

# The browser provides GPS to Streamlit.
# The WebRTC video processor can then read the latest
# coordinates from this dictionary.

LATEST_GPS = {
    "latitude": None,
    "longitude": None,
    "timestamp": None
}


# =========================================================
# GPS FUNCTION
# =========================================================

def get_gps_location():

    """
    Get the current browser/device location.

    This uses the browser's Geolocation API rather than
    IP-based location.
    """

    location = streamlit_js_eval(
        js_expressions="""
        new Promise((resolve) => {

            if (!navigator.geolocation) {

                resolve({
                    error: "Geolocation is not supported by this browser."
                });

                return;
            }

            navigator.geolocation.getCurrentPosition(

                (position) => {

                    resolve({
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude,
                        accuracy: position.coords.accuracy
                    });

                },

                (error) => {

                    resolve({
                        error: error.message
                    });

                },

                {
                    enableHighAccuracy: true,
                    timeout: 30000,
                    maximumAge: 30000
                }
            );
        })
        """,
        key="driver_gps"
    )

    return location


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

st.sidebar.header("Detection Settings")

EAR_THRESH = st.sidebar.slider(
    "EAR Threshold (Eye Closure)",
    0.1,
    0.4,
    0.2,
    0.01
)

MAR_THRESH = st.sidebar.slider(
    "MAR Threshold (Yawning)",
    0.4,
    1.0,
    0.6,
    0.05
)

YAW_THRESH = st.sidebar.slider(
    "Yaw Threshold (Looking Away L/R)",
    10,
    50,
    15,
    1
)

PITCH_THRESH = st.sidebar.slider(
    "Pitch Threshold (Looking Down/Up)",
    10,
    40,
    20,
    1
)

DROWSY_FRAMES = st.sidebar.slider(
    "Frames to trigger Drowsy",
    10,
    60,
    30,
    1
)


# =========================================================
# EMERGENCY CONTACT
# =========================================================

st.sidebar.header("🚨 Emergency Settings")

emergency_contact = st.sidebar.text_input(
    "Emergency Contact",
    value="+919361010422"
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    # YOLOv8 Nano for phone detection
    yolo_model = YOLO("yolov8n.pt")

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained BiLSTM
    lstm_model = get_model(
        "drowsiness_lstm.pt",
        device=device
    )

    # Load training normalization values
    normalization = torch.load(
        "drowsiness_lstm_normalization.pt",
        map_location=device,
        weights_only=True
    )

    mean = normalization["mean"].to(device)
    std = normalization["std"].to(device)

    return (
        yolo_model,
        lstm_model,
        mean,
        std,
        device
    )


yolo_model, lstm_model, lstm_mean, lstm_std, device = load_models()


# =========================================================
# MEDIAPIPE
# =========================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)


# =========================================================
# WEBRTC
# =========================================================

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class ADASVideoProcessor(VideoTransformerBase):

    def __init__(self):

        # -------------------------------------------------
        # FRAME / DROWSINESS
        # -------------------------------------------------

        self.frame_counter = 0
        self.drowsy_counter = 0

        # 30-frame temporal sequence
        self.sequence_buffer = deque(maxlen=30)

        # -------------------------------------------------
        # STATE MACHINE
        # -------------------------------------------------

        self.state_machine = DriverStateMachine(
            drowsy_duration=2.0,
            critical_duration=10.0,
            emergency_duration=10.0
        )

        self.current_state = DriverState.NORMAL

        # -------------------------------------------------
        # RECOVERY
        # -------------------------------------------------

        self.recovery_counter = 0

        # Driver must appear normal for these many frames
        self.RECOVERY_FRAMES = 15

        # -------------------------------------------------
        # ALARM
        # -------------------------------------------------

        self.alarm_manager = AlarmManager()

        # -------------------------------------------------
        # EMERGENCY
        # -------------------------------------------------

        self.emergency_manager = EmergencyManager(
            emergency_contact=emergency_contact
        )

        # Prevent repeated emergency processing
        self.emergency_sent = False

        # -------------------------------------------------
        # ONE-SHOT VOICE ALERT STATES
        # -------------------------------------------------

        self.last_phone_state = False
        self.last_distraction_state = False

        # -------------------------------------------------
        # MODELS
        # -------------------------------------------------

        self.yolo_model = yolo_model
        self.lstm_model = lstm_model

        self.lstm_mean = lstm_mean
        self.lstm_std = lstm_std

        self.device = device

        self.face_mesh = face_mesh

        # -------------------------------------------------
        # YOLO CACHE
        # -------------------------------------------------

        self.phone_detected = False

        # -------------------------------------------------
        # LSTM PROBABILITY
        # -------------------------------------------------

        self.lstm_probability = 0.0

    # =====================================================
    # RECEIVE FRAME
    # =====================================================

    def recv(self, frame):

        # -------------------------------------------------
        # CONVERT FRAME
        # -------------------------------------------------

        img = frame.to_ndarray(format="bgr24")

        h, w, _ = img.shape

        self.frame_counter += 1

        # =================================================
        # 1. PHONE DETECTION - YOLO
        # =================================================

        # YOLO is expensive, so run it every 5 frames.
        if self.frame_counter % 5 == 0:

            results = self.yolo_model(
                img,
                classes=[67],
                verbose=False
            )

            detected = False

            for result in results:

                if result.boxes is not None:

                    for box in result.boxes:

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )

                        cv2.rectangle(
                            img,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            img,
                            "Phone Distraction!",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                        detected = True

            self.phone_detected = detected

        phone_detected = self.phone_detected

        # -------------------------------------------------
        # PHONE VOICE ALERT
        # -------------------------------------------------

        if phone_detected and not self.last_phone_state:

            self.alarm_manager.speak_alert("phone")

        self.last_phone_state = phone_detected

        # =================================================
        # 2. MEDIAPIPE FACE PROCESSING
        # =================================================

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        fm_result = self.face_mesh.process(img_rgb)

        # -------------------------------------------------
        # Default values
        # -------------------------------------------------

        drowsy_alert = False
        lstm_drowsy = False
        looking_away = False

        ear = 0.0
        mar = 0.0

        pitch = 0.0
        yaw = 0.0
        roll = 0.0

        # =================================================
        # FACE FOUND
        # =================================================

        if fm_result.multi_face_landmarks:

            landmarks = fm_result.multi_face_landmarks[0]

            # -------------------------------------------------
            # Extract:
            # EAR
            # MAR
            # Pitch
            # Yaw
            # Roll
            # -------------------------------------------------

            ear, mar, pitch, yaw, roll = (
                extract_mediapipe_features(
                    landmarks,
                    w,
                    h
                )
            )

            # =================================================
            # HEAD POSE / DISTRACTION
            # =================================================

            if (
                abs(yaw) > YAW_THRESH
                or pitch > PITCH_THRESH
                or pitch < -PITCH_THRESH
            ):

                looking_away = True

                cv2.putText(
                    img,
                    "DISTRACTION: Looking Away",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    3
                )

            # -------------------------------------------------
            # DISTRACTION VOICE
            # -------------------------------------------------

            if (
                looking_away
                and not self.last_distraction_state
            ):

                self.alarm_manager.speak_alert(
                    "distraction"
                )

            self.last_distraction_state = looking_away

            # =================================================
            # RULE-BASED DROWSINESS
            # =================================================

            if (
                ear < EAR_THRESH
                or mar > MAR_THRESH
            ):

                self.drowsy_counter += 1

            else:

                self.drowsy_counter = 0

            if self.drowsy_counter >= DROWSY_FRAMES:

                drowsy_alert = True

            # =================================================
            # LSTM TEMPORAL FEATURES
            # =================================================

            current_features = np.array(
                [
                    ear,
                    mar,
                    pitch,
                    yaw,
                    roll
                ],
                dtype=np.float32
            )

            self.sequence_buffer.append(
                current_features
            )

            # -------------------------------------------------
            # LSTM prediction
            # -------------------------------------------------

            if len(self.sequence_buffer) == 30:

                sequence = np.array(
                    self.sequence_buffer,
                    dtype=np.float32
                )

                # Temporal differences
                differences = np.diff(
                    sequence[:, :4],
                    axis=0,
                    prepend=sequence[:1, :4]
                )

                # 5 base + 4 temporal = 9
                sequence_9 = np.concatenate(
                    [
                        sequence,
                        differences
                    ],
                    axis=1
                )

                seq_tensor = torch.tensor(
                    sequence_9,
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                # Normalize
                seq_tensor = (
                    seq_tensor - self.lstm_mean
                ) / (
                    self.lstm_std + 1e-8
                )

                # Prediction
                with torch.no_grad():

                    output = self.lstm_model(
                        seq_tensor
                    )

                    probabilities = torch.softmax(
                        output,
                        dim=1
                    )[0]

                    self.lstm_probability = float(
                        probabilities[1].item()
                    )

                    # Class 1 = Drowsy
                    if self.lstm_probability > 0.60:

                        lstm_drowsy = True

            # =================================================
            # DISPLAY FACE METRICS
            # =================================================

            cv2.putText(
                img,
                f"EAR: {ear:.2f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"MAR: {mar:.2f}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"YAW: {yaw:.1f}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"PITCH: {pitch:.1f}",
                (180, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"LSTM: {self.lstm_probability:.2f}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # =================================================
        # FACE NOT FOUND
        # =================================================

        else:

            self.last_distraction_state = False

            self.recovery_counter = 0

        # =================================================
        # COMBINE DROWSINESS DETECTORS
        # =================================================

        drowsy_detected = (
            drowsy_alert
            or lstm_drowsy
        )

        # =================================================
        # RECOVERY DETECTION
        # =================================================

        if drowsy_detected:

            self.recovery_counter = 0

        else:

            if fm_result.multi_face_landmarks:

                self.recovery_counter += 1

            else:

                self.recovery_counter = 0

        driver_recovered = (
            self.recovery_counter
            >= self.RECOVERY_FRAMES
        )

        # =================================================
        # STATE MACHINE
        # =================================================

        previous_state = self.current_state

        self.current_state = (
            self.state_machine.update(
                drowsy_detected=drowsy_detected,
                driver_recovered=driver_recovered
            )
        )

        # =================================================
        # ALARM STATE CONTROL
        # =================================================

        # -------------------------------------------------
        # ALARM
        # -------------------------------------------------

        if self.current_state == DriverState.ALARM:

            if previous_state != DriverState.ALARM:

                self.alarm_manager.start_alarm(
                    scenario="drowsy"
                )

        # -------------------------------------------------
        # CRITICAL
        # -------------------------------------------------

        elif self.current_state == DriverState.CRITICAL:

            if previous_state != DriverState.CRITICAL:

                self.alarm_manager.stop_alarm()

                self.alarm_manager.start_critical_alarm()

        # -------------------------------------------------
        # EMERGENCY
        # -------------------------------------------------

        elif self.current_state == DriverState.EMERGENCY:

            if previous_state != DriverState.EMERGENCY:

                self.alarm_manager.stop_alarm()

                self.alarm_manager.start_emergency_alarm()

                # =========================================
                # EMERGENCY RESPONSE
                # =========================================

                if not self.emergency_sent:

                    print("🚨 EMERGENCY STATE REACHED")

                    # Get latest GPS
                    gps = LATEST_GPS

                    if (
                        gps["latitude"] is not None
                        and gps["longitude"] is not None
                    ):
                        self.emergency_manager.update_location(
                            gps["latitude"],
                            gps["longitude"]
                        )

                    # Send exactly ONE emergency alert
                    success = self.emergency_manager.send_emergency_alert()

                    if success:
                        self.emergency_sent = True

        # -------------------------------------------------
        # NORMAL
        # -------------------------------------------------

        elif self.current_state == DriverState.NORMAL:

            if previous_state != DriverState.NORMAL:

                self.alarm_manager.stop_alarm()

                # Reset emergency system
                self.emergency_manager.reset()

                self.emergency_sent = False

        # =================================================
        # DISPLAY STATE
        # =================================================

        state_text = self.current_state.value

        cv2.putText(
            img,
            f"STATE: {state_text}",
            (20, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # =================================================
        # DISPLAY DROWSINESS WARNINGS
        # =================================================

        if drowsy_alert:

            cv2.putText(
                img,
                "WARNING: RULE-BASED DROWSINESS",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3
            )

        if lstm_drowsy:

            cv2.putText(
                img,
                "WARNING: LSTM DROWSINESS",
                (20, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3
            )

        # =================================================
        # PHONE WARNING
        # =================================================

        if phone_detected:

            cv2.putText(
                img,
                "WARNING: PHONE USAGE",
                (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3
            )

        # =================================================
        # STATE-SPECIFIC WARNINGS
        # =================================================

        if self.current_state == DriverState.ALARM:

            cv2.putText(
                img,
                "ALARM: PLEASE WAKE UP!",
                (20, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

        elif self.current_state == DriverState.CRITICAL:

            cv2.putText(
                img,
                "CRITICAL: DRIVER NOT RESPONDING",
                (20, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3
            )

        elif self.current_state == DriverState.EMERGENCY:

            cv2.putText(
                img,
                "EMERGENCY: CONTACT REQUIRED",
                (20, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3
            )

        # =================================================
        # GPS OVERLAY
        # =================================================

        if (
            LATEST_GPS["latitude"] is not None
            and LATEST_GPS["longitude"] is not None
        ):

            cv2.putText(
                img,
                "GPS: ACTIVE",
                (20, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                img,
                "GPS: UNAVAILABLE",
                (20, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        # =================================================
        # RETURN FRAME
        # =================================================

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )





# =========================================================
# LIVE ADAS FEED
# =========================================================

st.subheader("📹 Live ADAS Feed")

webrtc_streamer(
    key="adas-stream",

    mode=WebRtcMode.SENDRECV,

    rtc_configuration=RTC_CONFIGURATION,

    video_processor_factory=ADASVideoProcessor,

    media_stream_constraints={
        "video": {
            "width": {
                "ideal": 1280
            },
            "height": {
                "ideal": 720
            },
            "aspectRatio": {
                "ideal": 16 / 9
            }
        },
        "audio": False
    },

    async_processing=True
)
# =========================================================
# GPS SECTION
# =========================================================

st.subheader("📍 Driver GPS Location")

st.info(
    "Allow location access in your browser when prompted. "
    "The system uses the device/browser GPS location for "
    "emergency response."
)

gps_location = get_gps_location()

if gps_location:

    if (
        "latitude" in gps_location
        and "longitude" in gps_location
    ):

        latitude = gps_location["latitude"]
        longitude = gps_location["longitude"]

        # Store globally so WebRTC processor can access it
        LATEST_GPS["latitude"] = latitude
        LATEST_GPS["longitude"] = longitude
        LATEST_GPS["timestamp"] = datetime.now().isoformat()

        st.success("🟢 GPS location active")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Latitude",
                f"{latitude:.6f}"
            )

        with col2:

            st.metric(
                "Longitude",
                f"{longitude:.6f}"
            )

        with col3:

            accuracy = gps_location.get(
                "accuracy",
                None
            )

            if accuracy is not None:

                st.metric(
                    "Accuracy",
                    f"{accuracy:.1f} m"
                )

            else:

                st.metric(
                    "Accuracy",
                    "N/A"
                )

        maps_url = (
            "https://www.google.com/maps/"
            f"search/?api=1&query={latitude},{longitude}"
        )

        st.markdown(
            f"[📍 Open current location in Google Maps]"
            f"({maps_url})"
        )

    elif "error" in gps_location:

        st.error(
            f"❌ GPS error: {gps_location['error']}"
        )

else:

    st.warning(
        "⏳ Waiting for browser GPS permission..."
    )

# =========================================================
# PROJECT EXPLANATION
# =========================================================

st.markdown(
    """
    ---

    ## 🧠 How the System Works

    ### 1. Drowsiness Detection

    The system uses two approaches:

    **Rule-Based Detection**
    - EAR → eye closure
    - MAR → yawning
    - Persistent abnormal values trigger drowsiness.

    **Temporal BiLSTM**
    - 30-frame sequence
    - EAR
    - MAR
    - Pitch
    - Yaw
    - Roll
    - Temporal changes in EAR, MAR, Pitch and Yaw
    - Total = 9 features

    The two detectors are combined to determine the
    driver's drowsiness state.

    ### 2. Distraction Detection

    **Head Pose**
    - Yaw → looking left/right
    - Pitch → looking up/down

    **YOLOv8**
    - Detects mobile-phone usage.
    - YOLO runs every 5 frames to reduce processing load.

    ### 3. Driver State Machine

    The system follows:

    `NORMAL → ALARM → CRITICAL → EMERGENCY`

    Recovery requires consecutive normal frames before
    returning to `NORMAL`.

    ### 4. Smart Alarm

    The alarm manager runs:

    - 🔊 Beeping on one thread
    - 🗣️ Voice alerts on another thread

    Therefore the beep and spoken warning can occur
    simultaneously without blocking video processing.

    ### 5. Emergency Escalation

    If the driver does not recover:

    `ALARM → CRITICAL → EMERGENCY`

    The emergency state triggers:

    - 📍 Browser/device GPS
    - 🗺️ Google Maps location
    - 📱 Configured emergency contact
    - 🚨 Emergency alert generation
    """
)