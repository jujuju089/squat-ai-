
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
import tempfile
import time
from pathlib import Path

# -----------------------------
# SETTINGS
# -----------------------------
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

st.set_page_config(
    page_title="AI Squat Coach",
    page_icon="🏋️",
    layout="wide"
)

# -----------------------------
# FOLDER STRUCTURE
# -----------------------------
BASE_DIR = Path(__file__).parent
VIDEO_DIR = BASE_DIR / "videos"
RESULT_DIR = BASE_DIR / "results"

VIDEO_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🏋️ AI Squat Coach")
st.write("Lokale stabile Version mit Videospeicherung")

# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# -----------------------------
# HELPERS
# -----------------------------
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(
        c[1] - b[1], c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )

    angle = np.abs(radians * 180 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def save_uploaded_file(uploaded_file):
    save_path = VIDEO_DIR / uploaded_file.name

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(save_path)


# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Lade dein Squat Video hoch",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file:

    st.success("✅ Video erfolgreich hochgeladen")

    # Datei lokal speichern
    video_path = save_uploaded_file(uploaded_file)

    st.info(f"💾 Lokal gespeichert: {video_path}")

    # Start Button
    if st.button("▶ Analyse starten"):

        pose = load_pose()

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("❌ Video konnte nicht geöffnet werden")
            st.stop()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        stframe = st.empty()
        progress = st.progress(0)

        col1, col2, col3 = st.columns(3)

        reps_box = col1.empty()
        knee_box = col2.empty()
        feedback_box = col3.empty()

        reps = 0
        stage = None
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Jeder 2. Frame -> schneller
            if frame_count % 2 != 0:
                continue

            progress.progress(
                min(frame_count / total_frames, 1.0)
            )

            frame = cv2.resize(frame, (640, 360))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)

            feedback = "Suche Pose..."
            knee_angle = 0

            if results.pose_landmarks:

                lm = results.pose_landmarks.landmark

                hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
                ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

                knee_angle = calculate_angle(
                    hip, knee, ankle
                )

                back_angle = calculate_angle(
                    shoulder, hip, knee
                )

                # Rep Counter
                if knee_angle < 95:
                    stage = "down"

                if knee_angle > 160 and stage == "down":
                    reps += 1
                    stage = "up"

                # Feedback
                if knee_angle < 95:
                    depth = "Tief 👍"
                elif knee_angle < 120:
                    depth = "Okay 👍"
                else:
                    depth = "Zu flach ❌"

                if back_angle > 145:
                    back = "Rücken gut 👍"
                else:
                    back = "Rücken rund ⚠️"

                feedback = f"{depth} | {back}"

                # Draw Pose
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # Overlay
            cv2.putText(
                frame,
                f"Reps: {reps}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Knee: {int(knee_angle)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                feedback,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # UI Boxes
            reps_box.metric("Reps", reps)
            knee_box.metric("Knie Winkel", int(knee_angle))
            feedback_box.write(feedback)

            stframe.image(frame, channels="BGR")

        cap.release()

        st.success(f"✅ Analyse fertig! Wiederholungen: {reps}")

else:
    st.info("⬆️ Bitte Video hochladen")
