import streamlit as st
import cv2
import time
import os
import numpy as np
from detection import EnhancedObstacleDetector

# Page Configuration
st.set_page_config(
    page_title="AssistVision",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ AssistVision - Obstacle Detection System")
st.caption("Real-time object detection with distance estimation")

# Session State
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None

# Sidebar Controls
with st.sidebar:
    st.header("Configuration")
    video_source = st.radio(
        "Video Source",
        ("Webcam", "DroidCam", "Video File"),
        index=0
    )
    
    if video_source == "Video File":
        video_file = st.file_uploader("Upload video", type=["mp4", "avi"])
    else:
        camera_index = st.number_input(
            "Camera Index", 
            min_value=0, 
            max_value=10, 
            value=1 if video_source == "DroidCam" else 0
        )
    
    detection_thresh = st.slider(
        "Detection Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5
    )
    
    max_display_distance = st.slider(
        "Max Display Distance (m)", 
        min_value=1, 
        max_value=20, 
        value=10
    )

# Main Interface
col1, col2 = st.columns(2)
with col1:
    start_btn = st.button("Start System", type="primary")
with col2:
    stop_btn = st.button("Stop System")

video_placeholder = st.empty()
status_placeholder = st.empty()
info_placeholder = st.empty()

def initialize_video_source():
    if video_source == "Video File" and video_file:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.getbuffer())
        cap = cv2.VideoCapture(temp_path)
    else:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    return cap if cap.isOpened() else None

if start_btn:
    try:
        status_placeholder.info("Initializing system...")
        
        st.session_state.detector = EnhancedObstacleDetector(
            detection_thresh=detection_thresh,
            max_display_distance=max_display_distance
        )
        
        st.session_state.video_capture = initialize_video_source()
        
        if st.session_state.video_capture:
            st.session_state.is_running = True
            status_placeholder.success("System started successfully!")
        else:
            status_placeholder.error("Failed to initialize video source")
            
    except Exception as e:
        status_placeholder.error(f"Initialization error: {str(e)}")

if stop_btn:
    st.session_state.is_running = False
    if st.session_state.video_capture:
        st.session_state.video_capture.release()
    status_placeholder.info("System stopped")
    time.sleep(1)
    st.rerun()

# Main processing loop
if st.session_state.is_running and st.session_state.detector:
    cap = st.session_state.video_capture
    detector = st.session_state.detector
    start_time = time.time()
    
    while st.session_state.is_running:
        ret, frame = cap.read()
        if not ret:
            if video_source == "Video File":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                status_placeholder.warning("Video feed interrupted")
                break

        # Process frame
        processed_frame, detections = detector.process_frame(frame)
        
        # Display
        video_placeholder.image(
            cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_column_width=True
        )
        
        # Performance info
        elapsed = time.time() - start_time
        fps = detector.frame_count / elapsed if elapsed > 0 else 0
        info_placeholder.text(f"FPS: {fps:.1f} | Objects detected: {len(detections)}")
        
    # Cleanup
    if video_source == "Video File":
        try:
            os.remove("temp_video.mp4")
        except:
            pass
