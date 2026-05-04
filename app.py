import streamlit as st
import cv2
import numpy as np
from mediapipe import solutions
import tempfile
import os

st.set_page_config(page_title="Squat Analyzer", layout="wide", page_icon="🏋️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0a; color: #f0f0f0; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; color: #f0f0f0 !important; }
.hero-title { font-family: 'Bebas Neue', sans-serif; font-size: 5rem; letter-spacing: 6px; line-height: 1; color: #f0f0f0; margin: 0; }
.hero-sub { font-size: 0.85rem; letter-spacing: 4px; text-transform: uppercase; color: #C8F04A; margin-bottom: 0.5rem; }
.accent { color: #C8F04A; }
.section-title { font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; letter-spacing: 3px; color: #f0f0f0; border-bottom: 1px solid #222; padding-bottom: 0.5rem; margin: 2rem 0 1rem; }
.error-card { background: #111; border: 0.5px solid #2a2a2a; border-left: 3px solid #E24B4A; border-radius: 0 12px 12px 0; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
.error-card.good { border-left-color: #639922; }
.error-title { font-family: 'Bebas Neue', sans-serif; font-size: 1.2rem; letter-spacing: 2px; color: #E24B4A; margin-bottom: 0.5rem; }
.error-title.good { color: #639922; }
.error-text { font-size: 0.875rem; color: #888; line-height: 1.6; margin-bottom: 0.5rem; }
.tip-text { font-size: 0.875rem; color: #C8F04A; line-height: 1.6; }
.tip-label { font-size: 0.65rem; letter-spacing: 3px; text-transform: uppercase; color: #555; margin-bottom: 0.2rem; }
.rep-card { background: #111; border: 0.5px solid #2a2a2a; border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; }
.rep-card.worst { border: 1.5px solid #E24B4A; }
.rep-card.best { border: 1.5px solid #639922; }
.freq-bar-bg { background: #1a1a1a; border-radius: 4px; height: 6px; margin-top: 0.4rem; }
.freq-bar { background: #E24B4A; border-radius: 4px; height: 6px; }
.stProgress > div > div { background: #C8F04A !important; }
.stDownloadButton > button { background: #C8F04A !important; color: #0a0a0a !important; font-weight: 600 !important; border: none !important; border-radius: 8px !important; padding: 0.75rem 2rem !important; width: 100% !important; }
.stFileUploader > div { background: #111 !important; border: 1.5px dashed #333 !important; border-radius: 16px !important; }
[data-testid="metric-container"] { background: #111; border: 0.5px solid #222; border-radius: 12px; padding: 1rem; }
[data-testid="metric-container"] label { color: #666 !important; font-size: 0.7rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'Bebas Neue', sans-serif !important; font-size: 2rem !important; color: #C8F04A !important; }
</style>
""", unsafe_allow_html=True)

mp_pose = solutions.pose
mp_drawing = solutions.drawing_utils

FEHLER_ERKLAERUNGEN = {
    "ZU TIEF": {
        "erklaerung": "Du gehst unter 70° Kniewinkel — das erzeugt unnötigen Druck auf Kniegelenke und Menisken.",
        "tipp": "Stoppe wenn Oberschenkel parallel zum Boden sind (~90°). Stell dir vor du setzt dich auf einen Stuhl."
    },
    "ZU WENIG TIEFE": {
        "erklaerung": "Kniewinkel über 120° — Gesäß und Oberschenkel werden kaum aktiviert.",
        "tipp": "Geh tiefer bis Oberschenkel parallel zum Boden. Hüftmobilität mit täglichen Dehnübungen verbessern."
    },
    "OBERKÖRPER ZU WEIT VORNE": {
        "erklaerung": "Dein Torso kippt zu stark nach vorne — der untere Rücken übernimmt die Last.",
        "tipp": "Brust hoch, Blick geradeaus. Füße etwas breiter stellen. Planks für die Körpermitte trainieren."
    },
    "KNIE ZU WEIT VORNE": {
        "erklaerung": "Knie wandern weit über Fußspitzen — erhöhter Druck auf Knieknorpel und Bänder.",
        "tipp": "Gesäß aktiv nach hinten schieben. Knie nach außen drücken. Ferse fest am Boden halten."
    },
    "RÜCKEN NICHT GERADE": {
        "erklaerung": "Rundrücken unter Last ist gefährlich für Bandscheiben und Wirbelsäule.",
        "tipp": "Bauch vor jedem Rep anspannen, Schultern zurückziehen. Rudern und Face Pulls für den Rücken."
    },
    "ASYMMETRIE": {
        "erklaerung": "Linkes und rechtes Knie beugen sich unterschiedlich stark — deutet auf muskuläre Dysbalance hin.",
        "tipp": "Einbeinige Übungen (Bulgaren Split Squat) trainieren. Schwächere Seite zuerst üben."
    }
}

@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = load_pose()

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def analyze_squat(l_knee, r_knee, hip_angle, l_knee_x, l_ankle_x, r_knee_x, r_ankle_x, shoulder_angle):
    errors = []
    avg_knee = (l_knee + r_knee) / 2
    if avg_knee < 70:
        errors.append(("ZU TIEF", (0, 0, 220)))
    elif 120 < avg_knee < 160:
        errors.append(("ZU WENIG TIEFE", (0, 140, 255)))
    if hip_angle < 55:
        errors.append(("OBERKÖRPER ZU WEIT VORNE", (0, 0, 220)))
    if l_knee_x > l_ankle_x + 0.08 or r_knee_x > r_ankle_x + 0.08:
        errors.append(("KNIE ZU WEIT VORNE", (0, 0, 220)))
    if shoulder_angle < 160:
        errors.append(("RÜCKEN NICHT GERADE", (0, 0, 220)))
    if abs(l_knee - r_knee) > 15:
        errors.append(("ASYMMETRIE", (0, 120, 255)))
    if not errors:
        errors.append(("GUTE FORM", (100, 200, 50)))
    return errors

# Header
st.markdown('<p class="hero-sub">KI-gestützte Bewegungsanalyse</p>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">SQUAT<span class="accent">.</span>AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#555;font-size:0.9rem;margin-top:0.5rem;margin-bottom:2rem;">Lade ein Video hoch — die KI analysiert beide Körperseiten, erkennt Fehler und hebt schwache Reps hervor.</p>', unsafe_allow_html=True)

# Einstellungen
with st.expander("⚙️ Einstellungen"):
    fps_limit = st.slider("FPS-Limit (niedrigere Werte = schnellere Verarbeitung)", 5, 30, 15)
    st.caption("Empfehlung: 10–15 FPS für normale Analyse, 30 für maximale Genauigkeit")

file = st.file_uploader("Video hochladen (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

if file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(original_fps / fps_limit))

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, min(original_fps, fps_limit), (640, 360))

    st.markdown('<p class="section-title">LIVE ANALYSE</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])

    with col2:
        rep_display = st.empty()
        angle_l_display = st.empty()
        angle_r_display = st.empty()
        hip_display = st.empty()
        st.markdown("---")
        status_display = st.empty()
        progress_bar = st.progress(0)
    with col1:
        stframe = st.empty()

    reps = 0
    stage = None
    frame_count = 0
    l_knee_angles = []
    r_knee_angles = []
    hip_angles = []
    all_errors = {}

    # Rep-Tracking
    rep_data = []
    current_rep_errors = []
    current_rep_min_knee = 180
    current_rep_start = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # FPS-Limiter
        if frame_count % frame_skip != 0:
            continue

        progress_bar.progress(min(frame_count / total_frames, 1.0))

        frame = cv2.resize(frame, (640, 360))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Beide Seiten laden
            l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            l_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
            l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]

            r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
            r_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
            shoulder_angle = calculate_angle(l_ear, l_shoulder, l_hip)
            avg_knee = (l_knee_angle + r_knee_angle) / 2

            l_knee_angles.append(l_knee_angle)
            r_knee_angles.append(r_knee_angle)
            hip_angles.append(hip_angle)

            # Rep Counter
            if avg_knee < 90:
                stage = "UNTEN"
                current_rep_min_knee = min(current_rep_min_knee, avg_knee)

            if avg_knee > 160 and stage == "UNTEN":
                reps += 1
                stage = "OBEN"

                errors_this_rep = analyze_squat(
                    l_knee_angle, r_knee_angle, hip_angle,
                    l_knee.x, l_ankle.x, r_knee.x, r_ankle.x, shoulder_angle
                )
                error_names = [e[0] for e in errors_this_rep if e[0] != "GUTE FORM"]
                rep_data.append({
                    "rep": reps,
                    "min_knee": current_rep_min_knee,
                    "errors": error_names,
                    "score": len(error_names)
                })
                current_rep_min_knee = 180

            errors = analyze_squat(
                l_knee_angle, r_knee_angle, hip_angle,
                l_knee.x, l_ankle.x, r_knee.x, r_ankle.x, shoulder_angle
            )

            for err_text, _ in errors:
                if err_text != "GUTE FORM":
                    all_errors[err_text] = all_errors.get(err_text, 0) + 1

            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(200, 240, 74), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            # Winkel auf Frame
            cv2.putText(frame, f"L-KNIE {int(l_knee_angle)}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 240, 74), 2)
            cv2.putText(frame, f"R-KNIE {int(r_knee_angle)}", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 240, 74), 2)
            cv2.putText(frame, f"HUFTE {int(hip_angle)}", (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 240, 74), 2)
            cv2.putText(frame, f"REPS {reps}", (16, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            if stage:
                cv2.putText(frame, stage, (16, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 50), 2)

            y_pos = 170
            for err_text, err_color in errors:
                cv2.putText(frame, err_text, (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, err_color, 2)
                y_pos += 24

            rep_display.metric("Reps", reps)
            angle_l_display.metric("Links Knie", f"{int(l_knee_angle)}°")
            angle_r_display.metric("Rechts Knie", f"{int(r_knee_angle)}°")
            hip_display.metric("Hüfte", f"{int(hip_angle)}°")
            status_text = " · ".join([e[0] for e in errors if e[0] != "GUTE FORM"]) or "✓ Gute Form"
            status_display.markdown(f'<p style="font-size:0.75rem;color:#888;">{status_text}</p>', unsafe_allow_html=True)

        stframe.image(frame, channels="BGR", use_container_width=True)
        out.write(frame)

    cap.release()
    out.release()
    os.unlink(tfile.name)

    # Ergebnisse
    if l_knee_angles:
        gesamt_frames = max(len(l_knee_angles), 1)
        fehler_frames = sum(all_errors.values())
        fehler_quote = fehler_frames / gesamt_frames

        if fehler_quote < 0.1:
            note = "A+"
        elif fehler_quote < 0.25:
            note = "A"
        elif fehler_quote < 0.45:
            note = "B"
        elif fehler_quote < 0.65:
            note = "C"
        else:
            note = "D"

        st.markdown('<p class="section-title">ERGEBNIS</p>', unsafe_allow_html=True)
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Reps", reps)
        r2.metric("Ø Links", f"{int(np.mean(l_knee_angles))}°")
        r3.metric("Ø Rechts", f"{int(np.mean(r_knee_angles))}°")
        r4.metric("Ø Hüfte", f"{int(np.mean(hip_angles))}°")
        r5.metric("Note", note)

        # Rep Übersicht
        if rep_data:
            st.markdown('<p class="section-title">REP ÜBERSICHT</p>', unsafe_allow_html=True)
            worst_rep = max(rep_data, key=lambda x: x["score"])
            best_rep = min(rep_data, key=lambda x: x["score"])

            for r in rep_data:
                is_worst = r["rep"] == worst_rep["rep"] and r["score"] > 0
                is_best = r["rep"] == best_rep["rep"]
                card_class = "rep-card worst" if is_worst else ("rep-card best" if is_best else "rep-card")
                badge = " 👎 SCHLECHTESTE REP" if is_worst else (" 👍 BESTE REP" if is_best else "")
                fehler_text = ", ".join(r["errors"]) if r["errors"] else "Keine Fehler"
                farbe = "#E24B4A" if r["errors"] else "#639922"
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-family:'Bebas Neue',sans-serif;font-size:1.1rem;letter-spacing:2px;color:#f0f0f0;">
                            REP {r["rep"]}{badge}
                        </span>
                        <span style="font-size:0.8rem;color:#666;">Min. Knie: {int(r["min_knee"])}°</span>
                    </div>
                    <div style="margin-top:0.4rem;font-size:0.8rem;color:{farbe};">{fehler_text}</div>
                </div>
                """, unsafe_allow_html=True)

        # Fehleranalyse
        st.markdown('<p class="section-title">FEHLERANALYSE & TIPPS</p>', unsafe_allow_html=True)
        if all_errors:
            for err, count in sorted(all_errors.items(), key=lambda x: -x[1]):
                prozent = int((count / gesamt_frames) * 100)
                info = FEHLER_ERKLAERUNGEN.get(err, {})
                st.markdown(f"""
                <div class="error-card">
                    <div class="error-title">{err}</div>
                    <div class="tip-label">Häufigkeit</div>
                    <div style="font-size:0.8rem;color:#666;margin-bottom:0.3rem">{prozent}% der Frames</div>
                    <div class="freq-bar-bg"><div class="freq-bar" style="width:{min(prozent,100)}%"></div></div>
                    <div style="margin-top:1rem;">
                        <div class="tip-label">Was passiert</div>
                        <div class="error-text">{info.get('erklaerung', '')}</div>
                        <div class="tip-label">Tipp</div>
                        <div class="tip-text">→ {info.get('tipp', '')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="error-card good">
                <div class="error-title good">KEINE FEHLER ERKANNT</div>
                <div class="error-text">Perfekte Form auf beiden Seiten. Weiter so!</div>
            </div>
            """, unsafe_allow_html=True)

        # Download
        st.markdown('<p class="section-title">VIDEO HERUNTERLADEN</p>', unsafe_allow_html=True)
        with open(out_path, "rb") as f:
            st.download_button(
                label="⬇ Analysiertes Video herunterladen",
                data=f,
                file_name="squat_analyse.mp4",
                mime="video/mp4"
            )
        os.unlink(out_path)

else:
    st.markdown("""
    <div style="border:1.5px dashed #333;border-radius:16px;padding:3rem;text-align:center;background:#111;">
        <p style="font-family:'Bebas Neue',sans-serif;font-size:1.5rem;letter-spacing:3px;color:#333;margin:0">VIDEO HIER ABLEGEN</p>
        <p style="color:#444;font-size:0.8rem;margin-top:0.5rem">MP4 · MOV · AVI</p>
    </div>
    """, unsafe_allow_html=True)
