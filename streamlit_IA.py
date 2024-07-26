import streamlit as st
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av
import cv2
import queue
from typing import List, NamedTuple
from ultralytics import YOLO
import supervision as sv
import logging
import os
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

logger = logging.getLogger(__name__)

def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    try:
        token = client.tokens.create()
    except TwilioRestException as e:
        st.warning(
            f"Error occurred while accessing Twilio API. Fallback to a free STUN server from Google. ({e})"  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    return token.ice_servers

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray
    
model = YOLO('yolov8s.pt')
names = model.names
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), 
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), 
    (0, 128, 128), (128, 0, 128), (192, 192, 192), (128, 128, 128), (0, 0, 0), 
    (255, 165, 0), (75, 0, 130), (255, 192, 203), (245, 245, 220), (220, 20, 60)
]


result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def put_text(frame, texte, p, color):
    font = 1
    font_scale = 1.0
    thick = 1
    cv2.putText(frame, texte, p, font, font_scale, color, thick)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    results = model(image, verbose=False)[0]
    if results is not None:
        detections = sv.Detections.from_ultralytics(results)
        for detect in detections:
            num = int(detect[3])
            color = (0, 0, 0)
            if num < 20:
                color = colors[num]
            texte = names[num]
            x, y, w, h = np.array(detect[0]).astype(int)
            cv2.rectangle(image, (x, y), (w, h), color, 2)
            p = [x, y-20]
            put_text(image, texte, p, color=color)
            
        
    
    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
