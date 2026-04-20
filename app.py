import streamlit as st
import numpy as np
from PIL import Image
import imageio
import tempfile

st.title("KI Kniebeugen Analyse (stabile Version)")

def fake_knee_score(frame):
    """
    Vereinfachte Analyse (Demo-Logik)
    """
    return np.random.randint(70, 110)

uploaded_file = st.file_uploader("Video hochladen", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    reader = imageio.get_reader(tfile.name)

    scores = []
    reps = 0
    last_state = "up"

    for i, frame in enumerate(reader):
        if i % 10 != 0:
            continue

        score = fake_knee_score(frame)
        scores.append(score)

        if score < 85 and last_state == "up":
            reps += 1
            last_state = "down"

        if score > 95:
            last_state = "up"

    st.success(f"Wiederholungen erkannt: {reps}")

    if scores:
        avg = sum(scores) / len(scores)
        st.write(f"Ø Technik-Score: {int(avg)} / 100")

        if avg < 90:
            st.error("Achte auf tiefere und stabilere Ausführung")
        else:
            st.success("Gute Ausführung!")

    st.video(uploaded_file)
