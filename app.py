
import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp

# =========================
# CONFIG / NAVIGATION
# =========================
st.set_page_config(page_title="Fitness AI Coach", page_icon="🏋️")

page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Startseite", "🏋️ Übungen"]
)

# =========================
# STARTSEITE
# =========================
if page == "🏠 Startseite":
    st.title("🏋️ Fitness AI Coach")
    st.write("Willkommen! Wähle eine Übung aus und analysiere deine Technik mit KI.")

    st.info("👉 Aktuell verfügbar: Squats")
    st.warning("Bankdrücken & Kreuzheben sind noch in Arbeit 🚧")

# =========================
# ÜBUNGEN
# =========================
if page == "🏋️ Übungen":

    exercise = st.selectbox(
        "Übung auswählen",
        ["Squats", "Bankdrücken (in Arbeit)", "Kreuzheben (in Arbeit)"]
    )

    # =========================
    # SQUATS (AKTIV)
    # =========================
    if exercise == "Squats":
        st.header("🏋️ Squat Analyse")

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

        uploaded_file = st.file_uploader("📹 Squat Video hochladen", type=["mp4", "mov", "avi"])

        if uploaded_file is not None:

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)

            reps = 0
            stage = None
            angles = []

            st.info("Analyse läuft...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if results.pose_landmarks:

                    lm = results.pose_landmarks.landmark

                    hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    angle = calculate_angle(hip, knee, ankle)
                    angles.append(angle)

                    if angle < 90:
                        stage = "down"

                    if angle > 160 and stage == "down":
                        stage = "up"
                        reps += 1

            cap.release()

            st.success(f"🏋️ Wiederholungen: {reps}")

            if angles:
                avg = sum(angles) / len(angles)
                st.write(f"📊 Ø Kniewinkel: {int(avg)}°")

                if avg < 110:
                    st.error("⚠️ Geh tiefer in die Knie!")
                else:
                    st.success("✅ Gute Ausführung!")

            st.video(uploaded_file)

    # =========================
    # BANKDRÜCKEN
    # =========================
    elif exercise == "Bankdrücken (in Arbeit)":
        st.header("🏋️ Bankdrücken")
        st.info("🚧 Diese Funktion ist noch in Arbeit")

    # =========================
    # KREUZHEBEN
    # =========================
    elif exercise == "Kreuzheben (in Arbeit)":
        st.header("🏋️ Kreuzheben")
        st.info("🚧 Diese Funktion ist noch in Arbeit")
