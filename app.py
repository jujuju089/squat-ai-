import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os

# 🔥 verhindert MediaPipe Probleme
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

st.set_page_config(page_title="AI Squat Coach", layout="wide")

st.title("🏋️ AI Squat Coach")
st.write("Squat Analyse mit KI (stabile Version)")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ✅ MediaPipe nur einmal laden
@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0
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

# Upload
file = st.file_uploader("📤 Lade dein Squat Video hoch", type=["mp4", "mov", "avi"])

if file:
    pose = load_pose()

    # ✅ FIX: echtes File speichern (kein Permission Bug)
    video_path = "input_video.mp4"
    with open(video_path, "wb") as f:
        f.write(file.read())

    cap = cv2.VideoCapture(video_path)

    # ✅ Sicherheitscheck
    if not cap.isOpened():
        st.error("❌ Video konnte nicht geöffnet werden")
        st.stop()

    stframe = st.empty()

    reps = 0
    stage = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 Performance Fix
        frame_count += 1
        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (640, 360))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        feedback = ""

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
            ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

            # Winkel
            knee_angle = calculate_angle(hip, knee, ankle)
            back_angle = calculate_angle(shoulder, hip, knee)

            # REPS
            if knee_angle < 90:
                stage = "down"

            if knee_angle > 160 and stage == "down":
                reps += 1
                stage = "up"

            # Feedback
            if knee_angle < 90:
                depth = "Tief 👍"
            elif knee_angle < 120:
                depth = "Okay 👍"
            else:
                depth = "Zu flach ❌"

            if back_angle > 150:
                back = "Rücken gut 👍"
            else:
                back = "Rücken rund ⚠️"

            feedback = f"{depth} | {back}"

            # Overlay
            cv2.putText(frame, f"Reps: {reps}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, f"Knee: {int(knee_angle)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            cv2.putText(frame, feedback, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Skelett
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        stframe.image(frame, channels="BGR")

    cap.release()

    st.success(f"✅ Fertig! Wiederholungen: {reps}")

else:
    st.info("⬆️ Bitte lade ein Video hoch")
