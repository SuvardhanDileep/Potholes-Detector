import streamlit as st
import cv2
import tempfile
import time
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load YOLO model once
@st.cache_resource
def load_model():
    model = YOLO("yolov11n_.pt")  # replace with your trained model path
    return model

model = load_model()

st.title("🛣️ Pothole Detector")
st.markdown("Upload an **image**, **video**, or use **webcam** to detect potholes with YOLO!")

option = st.sidebar.radio("Choose Input Type:", ["Image", "Video", "Webcam"])

# ---------------- IMAGE UPLOAD ----------------
if option == "Image":
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img = Image.open(uploaded_img)
        img_array = np.array(img)
        img_array = cv2.resize(img_array, (320, 320))
        results = model(img_array)
        annotated_img = results[0].plot()

        st.image(annotated_img, caption="Detected Potholes", use_container_width=True)

# ---------------- VIDEO UPLOAD ----------------
elif option == "Video":
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count +=1
            if frame_count % 3!=0:
                continue


            frame = cv2.resize(frame, (320, 320))
            results = model(frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()

# ---------------- WEBCAM ----------------
elif option == "Webcam":
    st.markdown("Click start to use your webcam:")
    run = st.checkbox("Start Webcam")
    stframe = st.empty()

    cap = cv2.VideoCapture(0)

    frame_count = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count +=1
        if frame_count % 3!=0:
            continue

        frame = cv2.resize(frame, (320, 320))
        results = model(frame)
        annotated_frame = results[0].plot()

        stframe.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()
