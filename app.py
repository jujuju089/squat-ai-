import streamlit as st
import numpy as np
import mediapipe as mp
import imageio
from PIL import Image
import tempfile

st.title("KI Kniebeugen Analyse (Video Upload)")

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

    reader = imageio.get_reader(tfile.name)

    counter = 0
    stage = None
    angles = []

    with mp_pose.Pose() as pose:
        for i, frame in enumerate(reader):
            
            # nur jedes 5. Frame (schneller!)
            if i % 5 != 0:
                continue

            image = np.array(frame)

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

    st.success(f"Wiederholungen: {counter}")

    if angles:
        avg = sum(angles) / len(angles)
        st.write(f"Ø Kniewinkel: {int(avg)}°")

        if avg > 100:
            st.error("Gehe tiefer")
        else:
            st.success("Gute Tiefe")

    st.video(uploaded_file)
