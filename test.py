import streamlit as st
import cv2
import torch
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode

# Path to the YOLOv5 model weights
model_path = 'best.pt'

# Load the YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    return model

model = load_model()

# Video transformer using streamlit-webrtc
class YOLOv5VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Perform object detection
        results = self.model(img)
        img = np.squeeze(results.render())
        
        return img

# Streamlit app
def main():
    st.title("Real-Time Object Detection with YOLOv5")

    webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"}
        ]
    },
    video_transformer_factory=YOLOv5VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)


if __name__ == "__main__":
    main()
