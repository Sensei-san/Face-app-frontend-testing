import streamlit as st
from PIL import Image
import cv2
import numpy as np
import zipfile
import io
import json
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
POSES = [
    ("front", "Look straight at the camera"),
    ("left", "Turn your head slightly LEFT"),
    ("right", "Turn your head slightly RIGHT"),
    ("up", "Tilt your head UP"),
    ("down", "Tilt your head DOWN"),
]

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# HELPERS
# -----------------------------
def validate_single_face(image_pil):
    gray = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5)
    return len(faces) == 1


def overlay_face_outline(image_pil):
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape
    cv2.ellipse(
        img,
        (w // 2, h // 2),
        (int(w * 0.25), int(h * 0.35)),
        0,
        0,
        360,
        (0, 255, 0),
        2,
    )
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def build_zip(images, metadata):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for pose, img in images.items():
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            zipf.writestr(f"{pose}.jpg", img_bytes.getvalue())

        zipf.writestr("metadata.json", json.dumps(metadata, indent=2))

    buffer.seek(0)
    return buffer


# -----------------------------
# PAGE STATE
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "intro"

if "step" not in st.session_state:
    st.session_state.step = 0

if "images" not in st.session_state:
    st.session_state.images = {}

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Face Enrollment", layout="centered")
st.title("Employee Face Enrollment")

# -----------------------------
# PAGE 1 — INTRO
# -----------------------------
if st.session_state.page == "intro":
    name = st.text_input("Employee Name")
    emp_id = st.text_input("Employee ID")

    consent = st.checkbox(
        "I understand that images are collected for attendance system training purposes"
    )

    if st.button("➡ Start Enrollment"):
        if not (name and emp_id and consent):
            st.error("Please complete all fields and provide consent.")
        else:
            st.session_state.name = name
            st.session_state.emp_id = emp_id
            st.session_state.page = "capture"
            st.rerun()

    st.stop()

# -----------------------------
# PAGE 2 — CAPTURE
# -----------------------------
if st.session_state.page == "capture":
    step = st.session_state.step
    st.progress(step / len(POSES))

    pose_name, instruction = POSES[step]
    st.subheader(f"Step {step + 1}: {pose_name.upper()}")
    st.info(instruction)

    photo = st.camera_input("Align your face inside the outline")

    if photo:
        img = Image.open(photo)

        if not validate_single_face(img):
            st.error("Exactly one face must be visible. Please retake.")
            st.stop()

        preview = overlay_face_outline(img)
        st.image(preview, caption="Check alignment")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔁 Retake"):
                st.rerun()
        with col2:
            if st.button("✅ Accept & Continue"):
                st.session_state.images[pose_name] = img
                st.session_state.step += 1

                if st.session_state.step >= len(POSES):
                    st.session_state.page = "final"

                st.rerun()

    st.stop()

# -----------------------------
# PAGE 3 — FINAL
# -----------------------------
if st.session_state.page == "final":
    st.success("Enrollment complete!")

    metadata = {
        "employee_id": st.session_state.emp_id,
        "name": st.session_state.name,
        "poses": list(st.session_state.images.keys()),
        "timestamp": datetime.utcnow().isoformat(),
        "source": "streamlit-frontend-preview",
    }

    zip_buffer = build_zip(st.session_state.images, metadata)

    st.download_button(
        label="📥 Download Captured Data (ZIP)",
        data=zip_buffer,
        file_name=f"{st.session_state.emp_id}_{st.session_state.name}.zip",
        mime="application/zip",
    )

    st.info("This is a frontend-only preview. Backend upload is disabled.")
