import re
from pathlib import Path
import numpy as np
import cv2
import os
import pandas as pd
import mediapipe as mp

# Αρχικοποίηση της MediaPipe BlazePose
mp_pose = mp.solutions.pose

# Ορισμός των φακέλων που θα ψάξουμε για τα αρχεία .MOV
root_folders = [ Path("/home/poulimenos/project/KOA-PD-NM/KOA/KOA_EL"),
   Path("/home/poulimenos/project/KOA-PD-NM/KOA/KOA_MD"),
   Path("/home/poulimenos/project/KOA-PD-NM/KOA/KOA_SV"),
    Path("/home/poulimenos/project/KOA-PD-NM/PD/"),
    Path("/home/poulimenos/project/KOA-PD-NM/NM/")
]

# Εύρεση όλων των αρχείων .MOV
mov_files = [mov_file for folder in root_folders for mov_file in folder.rglob("*.MOV")]

# Φάκελος αποθήκευσης των αποτελεσμάτων
output_folder = Path("output")
output_folder.mkdir(parents=True, exist_ok=True)

# Επιθυμητά σημεία αναφοράς του σώματος
desired_landmarks = {
    'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP,
    'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP,
    'LEFT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE,
    'RIGHT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE,
    'LEFT_ELBOW': mp_pose.PoseLandmark.LEFT_ELBOW,
    'RIGHT_ELBOW': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'LEFT_WRIST': mp_pose.PoseLandmark.LEFT_WRIST,
    'RIGHT_WRIST': mp_pose.PoseLandmark.RIGHT_WRIST,
    'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
    'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE,
    'HEAD': mp_pose.PoseLandmark.NOSE
}

# Επεξεργασία κάθε αρχείου MOV
for mov_file in mov_files:
    cap = cv2.VideoCapture(str(mov_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    filename = os.path.basename(mov_file)

    # Έλεγχος αν το όνομα του αρχείου ακολουθεί το σωστό φορμά
    match = re.search(r"(\d{3})_(\w+)_(\d{2})", filename)
    if not match:
        print(f"Invalid filename format for {filename}")
        cap.release()
        continue

    # Ανάλυση του ονόματος του αρχείου για εξαγωγή πληροφοριών  +level gia koa,pd
    video_id, disease, side = match.groups()

    # Αρχικοποίηση του πίνακα για την αποθήκευση των δεδομένων πόζας
    pose_data = []

    # Επεξεργασία καρέ
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Επεξεργασία κάθε καρέ
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                frame_data = {
                    #'frame_time': cap.get(cv2.CAP_PROP_POS_MSEC),
                    'ID': video_id,
                    'Disease': disease,
                    'Side': side,
                    #'Level': level
                }

                # Προσθήκη δεδομένων από τα landmarks
                for name, landmark in desired_landmarks.items():
                    frame_data[f'{name}_x'] = landmarks[landmark].x
                    frame_data[f'{name}_y'] = landmarks[landmark].y
                    frame_data[f'{name}_z'] = landmarks[landmark].z
                    frame_data[f'{name}_visibility'] = landmarks[landmark].visibility

                pose_data.append(frame_data)

        
    # Αποθήκευση των δεδομένων σε CSV
    output_csv = output_folder / f"{video_id}_{disease}_{side}.csv"
    df = pd.DataFrame(pose_data)
    df.to_csv(output_csv, index=False)
    print(f"Processed {filename}, results saved to {output_csv}")

    cap.release()
