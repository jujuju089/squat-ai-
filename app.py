import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile

st.title("KI Kniebeugen Analyse (Video)")

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360 - angle
    return angle

uploaded_file = st.file_uploader("Lade ein Video hoch", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    counter = 0
    stage = None
    angles = []

    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            try:
                landmarks = results.pose_landmarks.landmark

                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)
                angles.append(angle)

                if angle > 160:
                    stage = "oben"
                if angle < 90 and stage == "oben":
                    stage = "unten"
                    counter += 1

            except:
                continue

    cap.release()

    st.success(f"Wiederholungen erkannt: {counter}")

    if len(angles) > 0:
        avg_angle = sum(angles) / len(angles)
        st.write(f"Durchschnittlicher Kniewinkel: {int(avg_angle)}°")

        if avg_angle > 100:
            st.error("Gehe tiefer in die Knie")
        else:
            st.success("Gute Tiefe")

    st.video(uploaded_file)
