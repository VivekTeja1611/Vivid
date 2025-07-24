import streamlit as st
import cv2
import time
import os
import numpy as np
from detection import EnhancedObstacleDetector
import threading
import queue
import traceback

# Page Configuration
st.set_page_config(
    page_title="Enhanced AssistVision",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 1rem;
}
.subtitle {
    font-size: 1.2rem;
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.status-success {
    color: #28a745;
    font-weight: bold;
}
.status-error {
    color: #dc3545;
    font-weight: bold;
}
.metrics-container {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.stButton > button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">👁️ Enhanced AssistVision</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Real-time Obstacle Detection for Blind Navigation</p>', unsafe_allow_html=True)

# Session State Initialization
def initialize_session_state():
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'video_capture' not in st.session_state:
        st.session_state.video_capture = None
    if 'frame_queue' not in st.session_state:
        st.session_state.frame_queue = queue.Queue(maxsize=5)
    if 'detection_stats' not in st.session_state:
        st.session_state.detection_stats = {
            'total_detections': 0,
            'danger_alerts': 0,
            'warning_alerts': 0,
            'frames_processed': 0,
            'fps': 0.0
        }
    if 'processing_thread' not in st.session_state:
        st.session_state.processing_thread = None
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = time.time()

initialize_session_state()

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Video Source Configuration
    st.subheader("Video Source")
    video_source_type = st.radio(
        "Source Type",
        ("Webcam", "DroidCam", "IP Camera", "Video File"),
        index=0,
        help="Select your video input source"
    )
    
    if video_source_type == "Video File":
        video_file = st.file_uploader(
            "Upload video", 
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a video file for processing"
        )
    elif video_source_type == "IP Camera":
        ip_url = st.text_input(
            "IP Camera URL", 
            placeholder="http://192.168.1.100:8080/video",
            help="Enter the IP camera stream URL"
        )
    else:
        camera_index = st.number_input(
            "Camera Index", 
            min_value=0, 
            max_value=10, 
            value=1 if video_source_type == "DroidCam" else 0,
            help="Camera device index (0 for default webcam)"
        )
    
    st.divider()
    
    # Detection Configuration
    st.subheader("Detection Settings")
    detection_thresh = st.slider(
        "Detection Confidence", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5,
        step=0.05,
        help="Minimum confidence threshold for object detection"
    )
    
    max_display_distance = st.slider(
        "Max Detection Distance (m)", 
        min_value=1, 
        max_value=20, 
        value=10,
        help="Maximum distance for object detection display"
    )
    
    # Network Configuration
    st.subheader("Phone Integration")
    tcp_port = st.number_input(
        "TCP Port",
        min_value=1000,
        max_value=9999,
        value=8888,
        help="TCP port for phone sensor data"
    )
    
    udp_port = st.number_input(
        "UDP Port", 
        min_value=1000,
        max_value=9999,
        value=8889,
        help="UDP port for phone sensor data"
    )
    
    # Safety Zone Configuration
    st.subheader("Safety Zones")
    enable_audio = st.checkbox(
        "Enable Audio Feedback", 
        value=True,
        help="Enable text-to-speech guidance"
    )
    
    guidance_frequency = st.slider(
        "Guidance Frequency (seconds)",
        min_value=1.0,
        max_value=5.0,
        value=1.5,
        step=0.5,
        help="Time interval between guidance messages"
    )
    
    st.divider()
    
    # Advanced Settings
    with st.expander("🔬 Advanced Settings"):
        show_debug_info = st.checkbox(
            "Show Debug Information",
            value=True,
            help="Display additional debugging information"
        )
        
        frame_skip = st.number_input(
            "Frame Skip",
            min_value=0,
            max_value=5,
            value=0,
            help="Skip frames for better performance (0 = no skip)"
        )

# Main Interface
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    start_btn = st.button(
        "🚀 Start Detection", 
        type="primary",
        disabled=st.session_state.is_running
    )

with col2:
    stop_btn = st.button(
        "⏹️ Stop Detection",
        disabled=not st.session_state.is_running
    )

with col3:
    manual_guidance_btn = st.button(
        "🔊 Manual Guidance",
        disabled=not st.session_state.is_running
    )

# Status and Video Display
status_placeholder = st.empty()
video_placeholder = st.empty()

# Statistics Display
if st.session_state.is_running:
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric(
            "FPS", 
            f"{st.session_state.detection_stats['fps']:.1f}",
            delta=None
        )
    
    with stats_col2:
        st.metric(
            "Objects Detected", 
            st.session_state.detection_stats['total_detections'],
            delta=None
        )
    
    with stats_col3:
        st.metric(
            "Danger Alerts", 
            st.session_state.detection_stats['danger_alerts'],
            delta=None
        )
    
    with stats_col4:
        st.metric(
            "Frames Processed", 
            st.session_state.detection_stats['frames_processed'],
            delta=None
        )

# Information Panel
info_placeholder = st.empty()

def initialize_video_source():
    """Initialize video capture based on selected source"""
    try:
        if video_source_type == "Video File" and 'video_file' in locals() and video_file:
            # Save uploaded file temporarily
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.getbuffer())
            cap = cv2.VideoCapture(temp_path)
        elif video_source_type == "IP Camera" and 'ip_url' in locals() and ip_url:
            cap = cv2.VideoCapture(ip_url)
        else:
            cap = cv2.VideoCapture(int(camera_index))
            # Set optimal camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        return cap if cap.isOpened() else None
    except Exception as e:
        st.error(f"Error initializing video source: {str(e)}")
        return None

def process_video_stream():
    """Process video stream in a separate thread"""
    if not st.session_state.video_capture or not st.session_state.detector:
        return
    
    cap = st.session_state.video_capture
    detector = st.session_state.detector
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while st.session_state.is_running:
            ret, frame = cap.read()
            if not ret:
                if video_source_type == "Video File":
                    # Loop video file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Error: Could not read frame")
                    break
            
            try:
                # Skip frames if configured
                if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                processed_frame, detections = detector.process_frame(frame)
                
                # Calculate FPS
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Update statistics
                st.session_state.detection_stats['frames_processed'] = frame_count
                st.session_state.detection_stats['total_detections'] = len(detections)
                st.session_state.detection_stats['fps'] = current_fps
                
                # Count safety zone alerts
                danger_count = 0
                warning_count = 0
                
                for det in detections:
                    zone_type = detector.safety_zone.get_zone_type(det.depth, det.velocity)
                    if zone_type in ["CRITICAL", "DANGER"]:
                        danger_count += 1
                    elif zone_type == "WARNING":
                        warning_count += 1
                
                st.session_state.detection_stats['danger_alerts'] = danger_count
                st.session_state.detection_stats['warning_alerts'] = warning_count
                
                # Add frame to queue for display
                if not st.session_state.frame_queue.full():
                    try:
                        # Clear old frames to prevent lag
                        while st.session_state.frame_queue.qsize() > 2:
                            st.session_state.frame_queue.get_nowait()
                        
                        st.session_state.frame_queue.put_nowait({
                            'frame': processed_frame,
                            'detections': detections,
                            'timestamp': time.time()
                        })
                    except queue.Full:
                        pass
                
                frame_count += 1
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
            except Exception as e:
                if show_debug_info:
                    print(f"Processing error: {str(e)}")
                    traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"Video processing thread error: {str(e)}")
        traceback.print_exc()
    finally:
        print("Video processing thread stopped")

# Button Actions
if start_btn:
    try:
        with status_placeholder:
            with st.spinner("Initializing Enhanced AssistVision..."):
                # Initialize detector with phone integration
                st.session_state.detector = EnhancedObstacleDetector(
                    detection_thresh=detection_thresh,
                    max_display_distance=max_display_distance,
                    tcp_port=tcp_port,
                    udp_port=udp_port
                )
                
                # Apply configuration
                if hasattr(st.session_state.detector, 'guidance_generator'):
                    st.session_state.detector.guidance_generator.guidance_interval = guidance_frequency
                
                # Start phone integration
                st.session_state.detector.start_phone_integration()
                
                # Initialize video source
                st.session_state.video_capture = initialize_video_source()
                
                if st.session_state.video_capture:
                    st.session_state.is_running = True
                    
                    # Start processing thread
                    st.session_state.processing_thread = threading.Thread(
                        target=process_video_stream,
                        daemon=True
                    )
                    st.session_state.processing_thread.start()
                    
                    st.success("✅ Enhanced AssistVision started successfully!")
                    
                    if enable_audio:
                        st.session_state.detector.audio_feedback.speak(
                            "Enhanced AssistVision started", priority=2
                        )
                        
                    # Display connection info
                    st.info(f"""
                    📱 **Phone Integration Active**
                    - TCP Port: {tcp_port}
                    - UDP Port: {udp_port}
                    - Connect your phone to this computer's IP address
                    """)
                else:
                    st.error("❌ Failed to initialize video source")
                    
    except Exception as e:
        st.error(f"❌ Initialization error: {str(e)}")
        if show_debug_info:
            st.code(traceback.format_exc())

if stop_btn:
    st.session_state.is_running = False
    
    # Stop phone integration
    if st.session_state.detector:
        st.session_state.detector.stop_phone_integration()
    
    # Release video capture
    if st.session_state.video_capture:
        st.session_state.video_capture.release()
        st.session_state.video_capture = None
    
    # Clear queues
    while not st.session_state.frame_queue.empty():
        try:
            st.session_state.frame_queue.get_nowait()
        except queue.Empty:
            break
    
    # Reset statistics
    st.session_state.detection_stats = {
        'total_detections': 0,
        'danger_alerts': 0,
        'warning_alerts': 0,
        'frames_processed': 0,
        'fps': 0.0
    }
    
    with status_placeholder:
        st.info("⏹️ Detection system stopped")
    
    if enable_audio and st.session_state.detector:
        st.session_state.detector.audio_feedback.speak(
            "Detection system stopped", priority=1
        )
    
    # Small delay before rerun
    time.sleep(0.5)
    st.rerun()

if manual_guidance_btn and st.session_state.detector:
    try:
        # Get latest frame data
        if not st.session_state.frame_queue.empty():
            frame_data = st.session_state.frame_queue.queue[-1]  # Get most recent
            detections = frame_data['detections']
            st.session_state.detector.manual_guidance(detections)
            st.success("🔊 Manual guidance triggered")
        else:
            st.warning("No current detections available")
            st.session_state.detector.audio_feedback.speak("No objects currently detected", priority=1)
    except Exception as e:
        st.error(f"Error triggering manual guidance: {str(e)}")

# Main video display and auto-refresh
if st.session_state.is_running:
    # Auto-refresh every 100ms for smooth video
    current_time = time.time()
    if current_time - st.session_state.last_update_time > 0.1:  # 10 FPS display update
        st.session_state.last_update_time = current_time
        
        # Display video stream
        if not st.session_state.frame_queue.empty():
            try:
                frame_data = st.session_state.frame_queue.get_nowait()
                processed_frame = frame_data['frame']
                detections = frame_data['detections']
                
                # Convert BGR to RGB for Streamlit
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                video_placeholder.image(
                    rgb_frame,
                    channels="RGB",
                    use_column_width=True,
                    caption=f"Enhanced AssistVision - {len(detections)} objects detected | FPS: {st.session_state.detection_stats['fps']:.1f}"
                )
                
                # Display detection information
                if detections and show_debug_info:
                    with info_placeholder:
                        st.subheader("🎯 Current Detections")
                        
                        # Create detection table
                        detection_data = []
                        for i, det in enumerate(detections[:5]):  # Show top 5 detections
                            if st.session_state.detector:
                                zone_type = st.session_state.detector.safety_zone.get_zone_type(
                                    det.depth, det.velocity
                                )
                            else:
                                zone_type = "UNKNOWN"
                            
                            detection_data.append({
                                "Object": det.label,
                                "Position": det.h_pos,
                                "Distance": f"{det.depth:.1f}m",
                                "Velocity": f"{det.velocity:.1f}m/s",
                                "Direction": det.direction,
                                "Safety Zone": zone_type
                            })
                        
                        if detection_data:
                            st.table(detection_data)
                        
                        # Phone sensor status
                        if st.session_state.detector:
                            imu_data = st.session_state.detector.network_server.get_imu_data()
                            phone_status = "🟢 Connected" if len(imu_data) > 0 else "🔴 Disconnected"
                            st.info(f"📱 Phone Sensor Status: {phone_status}")
                            
                            if len(imu_data) > 0:
                                st.success(f"📊 Receiving IMU data: {len(imu_data)} samples")
                            else:
                                st.warning("📱 Connect your phone to enhance navigation accuracy")
                        
            except queue.Empty:
                pass
            except Exception as e:
                if show_debug_info:
                    st.error(f"Display error: {str(e)}")
        
        # Auto-refresh for real-time updates
        time.sleep(0.1)
        st.rerun()

else:
    # Show system information when not running
    with video_placeholder:
        st.info("""
        ## 🎯 Enhanced AssistVision Features
        
        ### **For Blind Navigation**
        - 🚨 **Safety-First Design**: Conservative distance zones for maximum safety
        - 🔊 **Priority Audio Feedback**: Critical alerts override all other sounds
        - 🧭 **Precise Directional Guidance**: "slightly left", "directly ahead", etc.
        - 📱 **Phone Integration**: Uses your phone's sensors for 3D motion tracking
        - ⚡ **Real-time Collision Prediction**: Predicts potential hazards before they become dangerous
        
        ### **Advanced Technology**
        - **YOLOv8 Object Detection**: State-of-the-art real-time detection
        - **MiDaS Depth Estimation**: Neural network for accurate distance measurement
        - **DeepSORT Tracking**: Consistent object tracking across frames
        - **IMU Sensor Fusion**: 3D motion estimation from phone accelerometer/gyroscope
        
        ### 🚦 Safety Zone System
        - 🚨 **CRITICAL** (<0.8m): "STOP IMMEDIATELY!"
        - 🔴 **DANGER** (<1.5m): High priority alerts
        - 🟠 **WARNING** (<2.5m): Caution advised
        - 🟡 **CAUTION** (<4m): Be aware
        - 🟢 **SAFE** (>4m): Normal operation
        
        ### 📱 Phone Integration Setup
        1. Connect your phone to the same network as this computer
        2. Use the TCP/UDP ports shown above
        3. Send accelerometer and gyroscope data in JSON format
        4. Example apps: Sensor Logger, Physics Toolbox Suite
        
        **Configure your settings in the sidebar and click "Start Detection" to begin!**
        """)

# Network Status Display
if st.session_state.is_running and st.session_state.detector:
    with st.expander("📱 Network Status", expanded=False):
        imu_data = st.session_state.detector.network_server.get_imu_data()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("TCP Port", tcp_port)
            st.metric("IMU Samples", len(imu_data))
        
        with col2:
            st.metric("UDP Port", udp_port)
            connection_status = "Connected" if len(imu_data) > 0 else "Waiting"
            st.metric("Phone Status", connection_status)

# Cleanup function
if not st.session_state.is_running:
    # Clean up temporary video file
    if video_source_type == "Video File" and os.path.exists("temp_video.mp4"):
        try:
            os.remove("temp_video.mp4")
        except:
            pass

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Enhanced AssistVision - AI-Powered Navigation System for the Blind<br>"
    "Developed with ❤️ for accessibility and independence"
    "</div>", 
    unsafe_allow_html=True
)
