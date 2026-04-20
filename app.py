import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image

st.title("KI Kniebeugen Analyse")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360 - angle
    return angle

image_file = st.camera_input("Mach ein Bild deiner Kniebeuge")

if image_file is not None:
    image = Image.open(image_file)
    image_np = np.array(image)

    with mp_pose.Pose() as pose:
        results = pose.process(image_np)

        try:
            landmarks = results.pose_landmarks.landmark

            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)

            st.write(f"Kniewinkel: {int(angle)}°")

            if angle > 100:
                st.error("Gehe tiefer")
            else:
                st.success("Gute Tiefe")

        except:
            st.warning("Körper nicht erkannt")

    st.image(image)
