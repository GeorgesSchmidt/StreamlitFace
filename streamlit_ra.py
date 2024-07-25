import streamlit as st

from streamlit_webrtc import webrtc_streamer

import av
import cv2

from FaceAnalysis.detectLandmark import Detect

model = Detect()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    model.detect_on_image(img)
    flipped = img[::-1,:,:]

    return av.VideoFrame.from_ndarray(flipped, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
  