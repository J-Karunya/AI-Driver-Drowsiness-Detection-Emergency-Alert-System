import cv2
import numpy as np
from scipy.spatial import distance as dist

def calculate_ear(eye_points):
    """
    Compute the Eye Aspect Ratio (EAR).
    eye_points: list of (x, y) coordinates for the eye landmarks.
    """
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    # Compute the euclidean distance between the horizontal eye landmark
    C = dist.euclidean(eye_points[0], eye_points[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth_points):
    """
    Compute the Mouth Aspect Ratio (MAR).
    mouth_points: list of (x, y) coordinates for the mouth landmarks.
    """
    # Vertical distances
    A = dist.euclidean(mouth_points[13], mouth_points[19])
    B = dist.euclidean(mouth_points[14], mouth_points[18])
    C = dist.euclidean(mouth_points[15], mouth_points[17])
    # Horizontal distance
    D = dist.euclidean(mouth_points[12], mouth_points[16])
    # Compute MAR
    mar = (A + B + C) / (2.0 * D)
    return mar

def get_head_pose(face_2d, face_3d, img_w, img_h):
    """
    Calculate head pose using solvePnP.
    Returns angles (pitch, yaw, roll) in degrees.
    """
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    # Camera matrix
    focal_length = 1 * img_w
    cam_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Distortion coefficients
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)
    
    # Get angles
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    x = angles[0]
    y = angles[1]
    z = angles[2]

    return x, y, z  

def extract_mediapipe_features(landmarks, img_w, img_h):
    """
    Extracts EAR, MAR, and Head Pose angles from a MediaPipe normalized landmark list.
    """
    # MediaPipe Face Mesh landmark indices
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    MOUTH_INDICES = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
    ] # outer and inner lips
    
    # For Head Pose
    # Nose tip, Chin, Left eye left corner, Right eye right corner, Left mouth corner, Right mouth corner
    HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]
    
    # 3D generic face model (approximate)
    face_3d = []
    face_2d = []
    
    for idx, lm in enumerate(landmarks.landmark):
        if idx in HEAD_POSE_INDICES:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([lm.x, lm.y, lm.z])
            
    # Eye & Mouth coords
    left_eye = [(landmarks.landmark[i].x * img_w, landmarks.landmark[i].y * img_h) for i in LEFT_EYE_INDICES]
    right_eye = [(landmarks.landmark[i].x * img_w, landmarks.landmark[i].y * img_h) for i in RIGHT_EYE_INDICES]
    # Simple Inner Mouth for MAR
    inner_mouth = [(landmarks.landmark[i].x * img_w, landmarks.landmark[i].y * img_h) for i in [78, 81, 13, 311, 308, 402, 14, 178]] 
    # Using specific indices for MAR to match 8-point MAR calculation:
    # 0: left corner (78), 4: right corner (308)
    # 1,2,3: top inner lip (81, 13, 311), 5,6,7: bottom inner lip (315, 14, 87)
    mar_indices = [78, 81, 13, 311, 308, 315, 14, 87]
    mouth_points = [(landmarks.landmark[i].x * img_w, landmarks.landmark[i].y * img_h) for i in mar_indices]
    
    ear_left = calculate_ear(left_eye)
    ear_right = calculate_ear(right_eye)
    ear = (ear_left + ear_right) / 2.0
    
    # Custom MAR calculation for 8-point inner lip
    # Vertical: 81-87, 13-14, 311-315
    # Horizontal: 78-308
    A = dist.euclidean(mouth_points[1], mouth_points[7])
    B = dist.euclidean(mouth_points[2], mouth_points[6])
    C = dist.euclidean(mouth_points[3], mouth_points[5])
    D = dist.euclidean(mouth_points[0], mouth_points[4])
    mar = (A + B + C) / (2.0 * D) if D != 0 else 0
    
    pitch, yaw, roll = get_head_pose(face_2d, face_3d, img_w, img_h)
    
    return ear, mar, pitch, yaw, roll
