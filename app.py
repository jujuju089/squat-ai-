import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Squat Coach",
    page_icon="🏋️",
    layout="wide"
)

# =========================
# STYLING (UI schöner machen)
# =========================
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stMetric {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("🏋️ AI Squat Coach")
st.markdown("**Analyse deiner Kniebeugen mit KI (Pose Estimation)**")

# =========================
# MEDIAPIPE SETUP
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# =========================
# HELPER FUNCTIONS
# =========================
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

def get_landmarks(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pose.process(image)

# =========================
# FILE UPLOAD
# =========================
file = st.file_uploader("📤 Lade dein Squat Video hoch", type=["mp4", "mov", "avi"])

# =========================
# LAYOUT (2 SPALTEN)
# =========================
col1, col2 = st.columns([2, 1])

if file:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(file.read())

    cap = cv2.VideoCapture(tmp.name)

    stframe = col1.empty()

    reps = 0
    stage = None
    feedback = ""
    score = 0

    # KPIs
    rep_box = col2.empty()
    angle_box = col2.empty()
    feedback_box = col2.empty()
    score_box = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res = get_landmarks(frame)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
            ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

            # Winkel
            knee_angle = calculate_angle(hip, knee, ankle)
            back_angle = calculate_angle(shoulder, hip, knee)

            # REP LOGIK
            if knee_angle < 90:
                stage = "down"

            if knee_angle > 160 and stage == "down":
                reps += 1
                stage = "up"

            # FEEDBACK
            if knee_angle < 90:
                depth_feedback = "Perfekte Tiefe 🔥"
            elif knee_angle < 120:
                depth_feedback = "Gute Tiefe 👍"
            else:
                depth_feedback = "Zu flach ❌"

            if back_angle > 160:
                back_feedback = "Rücken stabil 👍"
            else:
                back_feedback = "Rücken beugt sich ⚠️"

            feedback = f"{depth_feedback} | {back_feedback}"

            # SCORE
            score = 0
            if knee_angle < 100:
                score += 50
            if back_angle > 150:
                score += 50

            # OVERLAY TEXT
            cv2.putText(frame, f"Reps: {reps}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, f"Knee: {int(knee_angle)}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            cv2.putText(frame, feedback, (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # SKELETT
            mp_drawing.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # KPI ANZEIGE
            rep_box.metric("Reps", reps)
            angle_box.metric("Knie Winkel", int(knee_angle))
            feedback_box.markdown(f"**Feedback:** {feedback}")
            score_box.metric("Score", score)

        stframe.image(frame, channels="BGR")

    cap.release()

    st.success(f"✅ Analyse fertig: {reps} Wiederholungen")

else:
    st.info("⬆️ Lade ein Video hoch, um zu starten")
