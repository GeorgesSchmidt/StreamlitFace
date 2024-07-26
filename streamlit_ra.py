import streamlit as st

from streamlit_webrtc import webrtc_streamer
import threading
import av
import cv2
from matplotlib import pyplot as plt
from FaceAnalysis.detectLandmark import Detect

lock = threading.Lock()
img_container = {"img": None}


model = Detect()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
        

    return img


ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

fig_place = st.empty()
fig, ax = plt.subplots(1, 1)

while ctx.state.playing:
    with lock:
        img = img_container["img"]
    if img is None:
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = model.detect_on_image(color_rgb)
    ax.cla()
    if frame is not None:
            ax.imshow(frame)
    
    #ax.hist(gray.ravel(), 256, [0, 256])
    fig_place.pyplot(fig)
  