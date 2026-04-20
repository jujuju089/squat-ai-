import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp

st.title("🏋️ KI Kniebeugen Analyse (MediaPipe Pose)")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle
    return angle


uploaded_file = st.file_uploader("Video hochladen", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    reps = 0
    stage = None
    knee_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        try:
            landmarks = results.pose_landmarks.landmark

            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)
            knee_angles.append(angle)

            # Squat Logik
            if angle < 90:
                stage = "down"
            if angle > 160 and stage == "down":
                stage = "up"
                reps += 1

        except:
            pass

    cap.release()

    st.success(f"🏋️ Wiederholungen erkannt: {reps}")

    if knee_angles:
        avg_angle = sum(knee_angles) / len(knee_angles)
        st.write(f"📊 Ø Knie-Winkel: {int(avg_angle)}°")

        if avg_angle < 110:
            st.error("⚠️ Tiefer in die Knie gehen!")
        else:
            st.success("✅ Gute Squat-Tiefe!")

    st.video(uploaded_file)
