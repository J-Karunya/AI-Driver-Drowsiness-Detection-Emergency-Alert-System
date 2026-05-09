import cv2
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from utils import extract_mediapipe_features
from drowsiness_lstm import DrowsinessLSTM

# Set to True if you want to skip feature extraction and just use random dummy data for structural demo
# Set to False to actually process the huge video files
DUMMY_TRAIN_FOR_DEMO = False 

def extract_features_from_video(video_path, label, sequence_length=30):
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5, 
        max_num_faces=1
    )
    cap = cv2.VideoCapture(video_path)
    
    sequences = []
    labels = []
    current_seq = []
    
    # We only process a max of 3000 frames per video to save time in demo
    frame_count = 0
    max_frames = 1000
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fm_result = mp_face_mesh.process(img_rgb)
        
        if fm_result.multi_face_landmarks:
            landmarks = fm_result.multi_face_landmarks[0]
            ear, mar, pitch, yaw, roll = extract_mediapipe_features(landmarks, w, h)
            current_seq.append([ear, mar])
            
            if len(current_seq) == sequence_length:
                sequences.append(current_seq)
                labels.append(label)
                # Slide window by 10 frames
                current_seq = current_seq[10:]
                
        frame_count += 1
        
    cap.release()
    return sequences, labels

def main():
    if DUMMY_TRAIN_FOR_DEMO:
        print("Using dummy data for training to generate the .pt file quickly...")
        X = torch.rand(100, 30, 2)
        y = torch.randint(0, 2, (100,))
    else:
        print("Extracting features from UTA-RLDD Fold1_part1...")
        dataset_path = "Fold1_part1/Fold1_part1"
        all_sequences = []
        all_labels = []
        
        # 0.mov = Alert (Label 0)
        # 5.mov = Low Vigilance (Label 1)
        # 10.mov = Drowsy (Label 1)
        for subject_dir in glob.glob(os.path.join(dataset_path, "*")):
            print(f"Processing {subject_dir}...")
            for video_file in glob.glob(os.path.join(subject_dir, "*.mov")):
                basename = os.path.basename(video_file).lower()
                if "0.mov" in basename:
                    label = 0
                elif "5.mov" in basename or "10.mov" in basename:
                    label = 1
                else:
                    continue
                
                print(f"Extracting {video_file} with label {label}")
                seqs, lbls = extract_features_from_video(video_file, label)
                all_sequences.extend(seqs)
                all_labels.extend(lbls)
                
        if len(all_sequences) == 0:
            print("No sequences found. Please check dataset path. Exiting.")
            return
            
        X = torch.tensor(all_sequences, dtype=torch.float32)
        y = torch.tensor(all_labels, dtype=torch.long)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    model = DrowsinessLSTM(input_size=2, hidden_size=32, num_layers=2, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "drowsiness_lstm.pt")
    print("Model saved to drowsiness_lstm.pt")

if __name__ == "__main__":
    main()
