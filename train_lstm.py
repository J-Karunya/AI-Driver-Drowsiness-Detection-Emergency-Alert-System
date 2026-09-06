import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "UTA-RLDD Face Cropped Video"

VIDEO_LENGTHS = ["len5"]

MAX_VIDEOS_PER_CLASS = 20

SEQUENCE_LENGTH = 30
WINDOW_STEP = 10

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.30

PATIENCE = 6

SEED = 42

CACHE_FILE = "lstm_features_len5_20videos_9features.pt"
MODEL_FILE = "drowsiness_lstm.pt"
NORMALIZATION_FILE = "drowsiness_lstm_normalization.pt"

# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

# ============================================================
# FEATURE FUNCTIONS
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [61, 291, 13, 14, 78, 308]

POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]


def euclidean(p1, p2):
    return np.linalg.norm(
        np.array([p1.x, p1.y]) -
        np.array([p2.x, p2.y])
    )


def calculate_ear(landmarks, eye_indices):

    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)

    horizontal = euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def calculate_mar(landmarks):

    top = landmarks[13]
    bottom = landmarks[14]

    left = landmarks[61]
    right = landmarks[291]

    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


def calculate_head_pose(landmarks, width, height):

    image_points = np.array([
        [
            landmarks[1].x * width,
            landmarks[1].y * height
        ],
        [
            landmarks[152].x * width,
            landmarks[152].y * height
        ],
        [
            landmarks[33].x * width,
            landmarks[33].y * height
        ],
        [
            landmarks[263].x * width,
            landmarks[263].y * height
        ],
        [
            landmarks[61].x * width,
            landmarks[61].y * height
        ],
        [
            landmarks[291].x * width,
            landmarks[291].y * height
        ]
    ], dtype=np.float64)

    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0]
    ], dtype=np.float64)

    focal_length = width

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    return pitch, yaw, roll


# ============================================================
# EXTRACT RAW FRAME FEATURES
# ============================================================

def extract_frame_features(frame, face_mesh):

    height, width = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark

    ear_left = calculate_ear(landmarks, LEFT_EYE)
    ear_right = calculate_ear(landmarks, RIGHT_EYE)

    ear = (ear_left + ear_right) / 2.0

    mar = calculate_mar(landmarks)

    pitch, yaw, roll = calculate_head_pose(
        landmarks,
        width,
        height
    )

    return np.array([
        ear,
        mar,
        pitch,
        yaw,
        roll
    ], dtype=np.float32)


# ============================================================
# ADD TEMPORAL FEATURES
# ============================================================

def create_temporal_features(sequence):

    """
    Input:
        [30, 5]

    Original:
        EAR, MAR, Pitch, Yaw, Roll

    Added:
        dEAR, dMAR, dPitch, dYaw

    Total:
        9 features
    """

    sequence = np.asarray(sequence, dtype=np.float32)

    differences = np.diff(
        sequence[:, :4],
        axis=0,
        prepend=sequence[0:1, :4]
    )

    combined = np.concatenate(
        [sequence, differences],
        axis=1
    )

    return combined


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path, label, face_mesh):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        features = extract_frame_features(
            frame,
            face_mesh
        )

        if features is not None:
            frames.append(features)

    cap.release()

    if len(frames) < SEQUENCE_LENGTH:
        return []

    sequences = []

    for start in range(
        0,
        len(frames) - SEQUENCE_LENGTH + 1,
        WINDOW_STEP
    ):

        sequence = frames[
            start:start + SEQUENCE_LENGTH
        ]

        temporal_sequence = create_temporal_features(
            sequence
        )

        sequences.append(
            (
                temporal_sequence,
                label
            )
        )

    return sequences


# ============================================================
# FIND VIDEOS
# ============================================================

def get_subject_videos():

    subjects = defaultdict(
        lambda: {
            0: [],
            1: []
        }
    )

    for length in VIDEO_LENGTHS:

        length_path = os.path.join(
            DATASET_ROOT,
            length
        )

        if not os.path.exists(length_path):
            continue

        for subject in sorted(
            os.listdir(length_path)
        ):

            subject_path = os.path.join(
                length_path,
                subject
            )

            if not os.path.isdir(subject_path):
                continue

            for folder, label in [
                ("0", 0),
                ("10", 1)
            ]:

                class_path = os.path.join(
                    subject_path,
                    folder
                )

                if not os.path.exists(class_path):
                    continue

                videos = []

                for file in os.listdir(class_path):

                    if file.lower().endswith(
                        (".mp4", ".avi", ".mov", ".mkv")
                    ):

                        videos.append(
                            os.path.join(
                                class_path,
                                file
                            )
                        )

                videos = sorted(videos)

                videos = videos[
                    :MAX_VIDEOS_PER_CLASS
                ]

                subjects[subject][label].extend(
                    videos
                )

    return subjects


# ============================================================
# SUBJECT-LEVEL SPLIT
# ============================================================

def split_subjects(subjects):

    subject_names = sorted(subjects.keys())

    rng = np.random.default_rng(SEED)

    rng.shuffle(subject_names)

    total = len(subject_names)

    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train_subjects = subject_names[:train_end]
    val_subjects = subject_names[
        train_end:val_end
    ]
    test_subjects = subject_names[val_end:]

    return (
        train_subjects,
        val_subjects,
        test_subjects
    )


# ============================================================
# EXTRACT DATA
# ============================================================

def extract_dataset():

    print("\n========================================")
    print(" DATASET EXTRACTION")
    print("========================================")

    subjects = get_subject_videos()

    print(f"Subjects found: {len(subjects)}")

    (
        train_subjects,
        val_subjects,
        test_subjects
    ) = split_subjects(subjects)

    print(
        f"Train subjects: {len(train_subjects)}"
    )

    print(
        f"Validation subjects: {len(val_subjects)}"
    )

    print(
        f"Test subjects: {len(test_subjects)}"
    )

    datasets = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects
    }

    result = {}

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        for split_name, split_subjects_list in datasets.items():

            X = []
            y = []

            print(
                f"\nProcessing {split_name.upper()}..."
            )

            for subject in split_subjects_list:

                for label in [0, 1]:

                    videos = subjects[
                        subject
                    ][label]

                    for video_path in videos:

                        sequences = process_video(
                            video_path,
                            label,
                            face_mesh
                        )

                        for sequence, seq_label in sequences:

                            X.append(sequence)
                            y.append(seq_label)

            if len(X) == 0:
                raise RuntimeError(
                    f"No sequences extracted for {split_name}"
                )

            X = torch.tensor(
                np.array(X),
                dtype=torch.float32
            )

            y = torch.tensor(
                y,
                dtype=torch.long
            )

            result[split_name] = (X, y)

            print(
                f"{split_name.capitalize()}: "
                f"{X.shape}"
            )

    torch.save(
        result,
        CACHE_FILE
    )

    print(
        f"\n✓ Feature cache saved:"
        f"\n  {CACHE_FILE}"
    )

    return result


# ============================================================
# LOAD / EXTRACT DATA
# ============================================================

if os.path.exists(CACHE_FILE):

    print("\n========================================")
    print(" LOADING FEATURE CACHE")
    print("========================================")

    print(
        f"Using: {CACHE_FILE}"
    )

    data = torch.load(
        CACHE_FILE,
        weights_only=False
    )

else:

    data = extract_dataset()


X_train, y_train = data["train"]
X_val, y_val = data["val"]
X_test, y_test = data["test"]


# ============================================================
# NORMALIZATION
# ============================================================

print("\n========================================")
print(" NORMALIZATION")
print("========================================")

mean = X_train.mean(
    dim=(0, 1)
)

std = X_train.std(
    dim=(0, 1)
)

std = torch.clamp(
    std,
    min=1e-6
)


def normalize(X):

    return (
        X - mean
    ) / std


X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)


torch.save(
    {
        "mean": mean,
        "std": std
    },
    NORMALIZATION_FILE
)

print(
    f"✓ Normalization saved:"
    f"\n  {NORMALIZATION_FILE}"
)


# ============================================================
# DATASET / DATALOADER
# ============================================================

train_dataset = torch.utils.data.TensorDataset(
    X_train,
    y_train
)

val_dataset = torch.utils.data.TensorDataset(
    X_val,
    y_val
)

test_dataset = torch.utils.data.TensorDataset(
    X_test,
    y_test
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

class DrowsinessLSTM(nn.Module):

    def __init__(
        self,
        input_size=9,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        num_classes=2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                32
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                32,
                num_classes
            )
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        return self.fc(last_output)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n========================================")
print(" MODEL TRAINING")
print("========================================")

print(
    f"Training device: {device}"
)

print(
    f"Input features: {X_train.shape[-1]}"
)

print(
    f"Sequence length: {SEQUENCE_LENGTH}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = torch.bincount(
    y_train,
    minlength=2
).float()

total = class_counts.sum()

class_weights = total / (
    2.0 *
    class_counts.clamp(min=1)
)

class_weights = class_weights.to(device)

print(
    "\nClass counts:"
)

print(
    f"Alert  : {int(class_counts[0])}"
)

print(
    f"Drowsy : {int(class_counts[1])}"
)

print(
    f"Class weights: "
    f"{class_weights.cpu().numpy()}"
)


# ============================================================
# CREATE MODEL
# ============================================================

model = DrowsinessLSTM(
    input_size=9,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    correct = 0
    total_samples = 0

    alert_correct = 0
    alert_total = 0

    drowsy_correct = 0
    drowsy_total = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            output = model(X)

            predictions = torch.argmax(
                output,
                dim=1
            )

            correct += (
                predictions == y
            ).sum().item()

            total_samples += y.size(0)

            alert_mask = y == 0
            drowsy_mask = y == 1

            alert_total += alert_mask.sum().item()

            drowsy_total += drowsy_mask.sum().item()

            alert_correct += (
                predictions[alert_mask] == 0
            ).sum().item()

            drowsy_correct += (
                predictions[drowsy_mask] == 1
            ).sum().item()

    accuracy = correct / max(
        total_samples,
        1
    )

    alert_accuracy = alert_correct / max(
        alert_total,
        1
    )

    drowsy_accuracy = drowsy_correct / max(
        drowsy_total,
        1
    )

    return (
        accuracy,
        alert_accuracy,
        drowsy_accuracy
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_accuracy = 0.0
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    correct = 0
    total_samples = 0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(X)

        loss = criterion(
            output,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += (
            loss.item() * y.size(0)
        )

        predictions = torch.argmax(
            output,
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total_samples += y.size(0)

    train_loss = (
        running_loss /
        max(total_samples, 1)
    )

    train_accuracy = (
        correct /
        max(total_samples, 1)
    )

    (
        val_accuracy,
        val_alert_accuracy,
        val_drowsy_accuracy
    ) = evaluate(val_loader)

    scheduler.step(
        val_accuracy
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )

    print(
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.3f}"
    )

    print(
        f"Val Acc: {val_accuracy:.3f} | "
        f"Val Alert: {val_alert_accuracy:.3f} | "
        f"Val Drowsy: {val_drowsy_accuracy:.3f}"
    )

    print(
        f"Learning Rate: {current_lr:.6f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_FILE
        )

        print(
            f"  ✓ Best model saved "
            f"(validation accuracy: "
            f"{val_accuracy:.3f})"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"  No improvement "
            f"({epochs_without_improvement}/"
            f"{PATIENCE})"
        )

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best validation model..."
)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )
)


# ============================================================
# FINAL TEST
# ============================================================

(
    test_accuracy,
    test_alert_accuracy,
    test_drowsy_accuracy
) = evaluate(test_loader)


print("\n========================================")
print(" FINAL TEST RESULTS")
print("========================================")

print(
    f"\nOverall Test Accuracy : "
    f"{test_accuracy:.3f}"
)

print(
    f"Alert Accuracy        : "
    f"{test_alert_accuracy:.3f}"
)

print(
    f"Drowsy Accuracy       : "
    f"{test_drowsy_accuracy:.3f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

model.eval()

confusion = torch.zeros(
    2,
    2,
    dtype=torch.int64
)

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(device)

        output = model(X)

        predictions = torch.argmax(
            output,
            dim=1
        ).cpu()

        for actual, predicted in zip(
            y,
            predictions
        ):

            confusion[
                actual,
                predicted
            ] += 1


print("\n========================================")
print(" CONFUSION MATRIX")
print("========================================")

print(
    "\n              Predicted"
)

print(
    "              Alert  Drowsy"
)

print(
    f"Actual Alert  "
    f"{confusion[0,0]:5d}  "
    f"{confusion[0,1]:5d}"
)

print(
    f"Actual Drowsy "
    f"{confusion[1,0]:5d}  "
    f"{confusion[1,1]:5d}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print(" TRAINING COMPLETE")
print("========================================")

print(
    f"\n✓ Model:"
    f"\n  {MODEL_FILE}"
)

print(
    f"\n✓ Feature cache:"
    f"\n  {CACHE_FILE}"
)

print(
    f"\n✓ Normalization:"
    f"\n  {NORMALIZATION_FILE}"
)