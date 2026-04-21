import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp

st.set_page_config(page_title="AI Squat Coach", layout="wide")

st.title("🏋️ AI Squat Coach")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 👉 Pose leichter machen (WICHTIG)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,   # weniger Last
    enable_segmentation=False
)

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

file = st.file_uploader("Video hochladen", type=["mp4", "mov", "avi"])

if file:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(file.read())

    cap = cv2.VideoCapture(tmp.name)

    stframe = st.empty()

    reps = 0
    stage = None

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 👉 PERFORMANCE FIX: nur jedes 3. Frame
        frame_count += 1
        if frame_count % 3 != 0:
            continue

        # 👉 Resize (sehr wichtig!)
        frame = cv2.resize(frame, (640, 360))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
            ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]

            knee_angle = calculate_angle(hip, knee, ankle)

            # REP LOGIK
            if knee_angle < 90:
                stage = "down"

            if knee_angle > 160 and stage == "down":
                reps += 1
                stage = "up"

            # Overlay
            cv2.putText(frame, f"Reps: {reps}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, f"Angle: {int(knee_angle)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        stframe.image(frame, channels="BGR")

    cap.release()

    st.success(f"Fertig! Reps: {reps}")
