import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp

st.set_page_config(page_title="AI Fitness Coach", page_icon="🏋️")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

page = st.sidebar.selectbox(
    "Übung auswählen",
    ["🏠 Home", "🏋️ Squats", "💪 Bankdrücken", "🏋️ Kreuzheben"]
)

# =========================
# HOME
# =========================
if page == "🏠 Home":
    st.title("🏋️ AI Fitness Coach")
    st.write("Analyse deiner Übungen mit Pose Estimation KI.")
    st.info("Model: MediaPipe Pose (alle Übungen basieren darauf)")

# =========================
# HELP FUNCTION
# =========================
def get_landmarks(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pose.process(image)

# =========================
# SQUATS
# =========================
if page == "🏋️ Squats":
    st.title("🏋️ Squat Analyse")

    file = st.file_uploader("Video hochladen", type=["mp4", "mov", "avi"])

    if file:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(file.read())

        cap = cv2.VideoCapture(tmp.name)

        reps = 0
        stage = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            res = get_landmarks(frame)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value].y
                knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y

                if knee < hip:
                    stage = "down"
                if knee > hip and stage == "down":
                    reps += 1
                    stage = "up"

        cap.release()

        st.success(f"🏋️ Squats: {reps} Wiederholungen")

# =========================
# BANKDRÜCKEN
# =========================
if page == "💪 Bankdrücken":
    st.title("💪 Bankdrücken Analyse")

    file = st.file_uploader("Video hochladen", type=["mp4", "mov", "avi"])

    if file:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(file.read())

        cap = cv2.VideoCapture(tmp.name)

        reps = 0
        stage = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            res = get_landmarks(frame)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

                # runter = Ellbogen unter Schulter
                if elbow > shoulder:
                    stage = "down"

                if elbow < shoulder and stage == "down":
                    reps += 1
                    stage = "up"

        cap.release()

        st.success(f"💪 Bankdrücken: {reps} Wiederholungen")

# =========================
# KREUZHEBEN
# =========================
if page == "🏋️ Kreuzheben":
    st.title("🏋️ Kreuzheben Analyse")

    file = st.file_uploader("Video hochladen", type=["mp4", "mov", "avi"])

    if file:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(file.read())

        cap = cv2.VideoCapture(tmp.name)

        reps = 0
        stage = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            res = get_landmarks(frame)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value].y
                knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y

                # Hüfte hoch/runter Bewegung
                if hip < knee:
                    stage = "up"

                if hip > knee and stage == "up":
                    reps += 1
                    stage = "down"

        cap.release()

        st.success(f"🏋️ Kreuzheben: {reps} Wiederholungen")
