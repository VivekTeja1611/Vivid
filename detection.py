import cv2
import torch
import math
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import threading
import queue
import time
import socket
import json
from collections import deque
import pyttsx3
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

# Suppress YOLO logs
LOGGER.setLevel("WARNING")

@dataclass
class IMUData:
    """Structure for IMU sensor data"""
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0

@dataclass
class MotionState:
    """3D motion state estimation"""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    confidence: float

class IMUProcessor:
    """Process IMU data for motion estimation"""
    
    def __init__(self):
        self.imu_history = deque(maxlen=100)
        self.motion_state = MotionState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            orientation=np.array([1, 0, 0, 0]),  # quaternion
            angular_velocity=np.zeros(3),
            confidence=0.0
        )
        
        # Kalman filter parameters
        self.dt = 1/30.0  # Assuming 30Hz update rate
        self.gravity = np.array([0, 0, -9.81])
        
        # Low-pass filter for accelerometer
        self.accel_filter_b, self.accel_filter_a = butter(2, 0.1, 'low')
        self.accel_buffer = deque(maxlen=10)
        self.gyro_buffer = deque(maxlen=10)
        
        # Bias estimation
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.bias_samples = 0
        self.bias_estimation_samples = 100
        
    def add_imu_data(self, imu_data: IMUData):
        """Add new IMU data and update motion estimation"""
        self.imu_history.append(imu_data)
        
        # Convert to numpy arrays
        accel = np.array([imu_data.accel_x, imu_data.accel_y, imu_data.accel_z])
        gyro = np.array([imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z])
        
        # Bias estimation during initial calibration
        if self.bias_samples < self.bias_estimation_samples:
            self.accel_bias = (self.accel_bias * self.bias_samples + accel) / (self.bias_samples + 1)
            self.gyro_bias = (self.gyro_bias * self.bias_samples + gyro) / (self.bias_samples + 1)
            self.bias_samples += 1
            return
        
        # Remove bias
        accel_corrected = accel - self.accel_bias
        gyro_corrected = gyro - self.gyro_bias
        
        # Add to buffers for filtering
        self.accel_buffer.append(accel_corrected)
        self.gyro_buffer.append(gyro_corrected)
        
        if len(self.accel_buffer) < 5:
            return
        
        # Apply low-pass filter
        accel_filtered = self._apply_filter(self.accel_buffer, self.accel_filter_b, self.accel_filter_a)
        
        # Update orientation using gyroscope
        self._update_orientation(gyro_corrected)
        
        # Remove gravity from acceleration
        gravity_world = self._rotate_vector(self.gravity, self.motion_state.orientation)
        accel_world = self._rotate_vector(accel_filtered, self.motion_state.orientation) - gravity_world
        
        # Update motion state
        self.motion_state.acceleration = accel_world
        self.motion_state.velocity += accel_world * self.dt
        self.motion_state.position += self.motion_state.velocity * self.dt
        self.motion_state.angular_velocity = gyro_corrected
        
        # Apply velocity damping to prevent drift
        self.motion_state.velocity *= 0.95
        
        # Calculate confidence based on data consistency
        self._update_confidence()
    
    def _apply_filter(self, buffer, b, a):
        """Apply low-pass filter to buffer data"""
        if len(buffer) < 5:
            return buffer[-1]
        
        data = np.array(buffer)
        filtered = filtfilt(b, a, data.T).T
        return filtered[-1]
    
    def _update_orientation(self, gyro):
        """Update orientation using gyroscope data"""
        # Convert angular velocity to quaternion update
        norm = np.linalg.norm(gyro)
        if norm > 0:
            axis = gyro / norm
            angle = norm * self.dt
            
            # Create rotation quaternion
            q_rot = np.array([
                np.cos(angle/2),
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2)
            ])
            
            # Apply rotation to current orientation
            self.motion_state.orientation = self._quaternion_multiply(
                self.motion_state.orientation, q_rot
            )
            
            # Normalize quaternion
            self.motion_state.orientation /= np.linalg.norm(self.motion_state.orientation)
    
    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _rotate_vector(self, vector, quaternion):
        """Rotate vector by quaternion"""
        # Convert quaternion to rotation matrix
        rotation = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        return rotation.apply(vector)
    
    def _update_confidence(self):
        """Update confidence based on data consistency"""
        if len(self.imu_history) < 10:
            self.motion_state.confidence = 0.0
            return
        
        # Calculate variance in recent measurements
        recent_accel = np.array([[d.accel_x, d.accel_y, d.accel_z] for d in list(self.imu_history)[-10:]])
        accel_variance = np.var(recent_accel, axis=0).mean()
        
        # Confidence inversely related to variance
        self.motion_state.confidence = max(0.0, min(1.0, 1.0 - accel_variance / 10.0))

class OpticalFlowProcessor:
    """Process optical flow for enhanced motion estimation"""
    
    def __init__(self):
        self.prev_frame = None
        self.prev_gray = None
        self.prev_points = None
        self.flow_history = deque(maxlen=10)
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Dense optical flow
        self.flow_magnitude_history = deque(maxlen=5)
        self.flow_direction_history = deque(maxlen=5)
        
    def process_frame(self, frame):
        """Process frame for optical flow estimation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_data = {}
    
        if self.prev_gray is not None and self.prev_points is not None:
            next_points, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params
            )
    
            if next_points is not None and status is not None:
                # Filter valid points
                good_new = next_points[status == 1]
                good_old = self.prev_points[status == 1]
    
                if len(good_new) > 0:
                    motion_vectors = good_new - good_old
                    magnitudes = np.linalg.norm(motion_vectors, axis=1)
                    directions = np.arctan2(motion_vectors[:,1], motion_vectors[:,0])
    
                    flow_data = {
                        'magnitude_mean': np.mean(magnitudes),
                        'magnitude_std': np.std(magnitudes),
                        'direction_mean': np.mean(directions),
                        'direction_std': np.std(directions),
                        'flow_vectors': motion_vectors,
                        'confidence': float(np.sum(status)) / len(status)
                    }
    
                    self.flow_magnitude_history.append(flow_data['magnitude_mean'])
                    self.flow_direction_history.append(flow_data['direction_mean'])
    
                    # Update previous points
                    self.prev_points = good_new.reshape(-1, 1, 2)
                else:
                    self.prev_points = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            else:
                self.prev_points = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        else:
            self.prev_points = cv2.goodFeaturesToTrack(gray, **self.feature_params)
    
        self.prev_gray = gray.copy()
        return flow_data
    
    def get_ego_motion(self):
        """Estimate ego motion from optical flow"""
        if len(self.flow_magnitude_history) < 3:
            return np.zeros(3), 0.0
        
        # Calculate velocity from flow magnitude
        recent_flow = list(self.flow_magnitude_history)[-3:]
        flow_velocity = np.mean(recent_flow)
        
        # Calculate direction from flow direction
        recent_directions = list(self.flow_direction_history)[-3:]
        mean_direction = np.mean(recent_directions)
        
        # Convert to 3D velocity vector (assuming forward motion)
        velocity_3d = np.array([
            flow_velocity * np.cos(mean_direction),
            flow_velocity * np.sin(mean_direction),
            0.0
        ])
        
        confidence = 1.0 - np.std(recent_flow) / (np.mean(recent_flow) + 0.001)
        
        return velocity_3d, max(0.0, min(1.0, confidence))

class NetworkServer:
    """TCP and UDP server for receiving phone sensor data"""
    
    def __init__(self, tcp_port=8888, udp_port=8889):
        self.tcp_port = tcp_port
        self.udp_port = udp_port
        self.running = False
        self.imu_queue = queue.Queue()
        
        # Server sockets
        self.tcp_socket = None
        self.udp_socket = None
        
        # Client connections
        self.tcp_clients = []
        
    def start_servers(self):
        """Start TCP and UDP servers"""
        self.running = True
        
        # Start TCP server thread
        tcp_thread = threading.Thread(target=self._tcp_server, daemon=True)
        tcp_thread.start()
        
        # Start UDP server thread
        udp_thread = threading.Thread(target=self._udp_server, daemon=True)
        udp_thread.start()
        
        print(f"Network servers started - TCP: {self.tcp_port}, UDP: {self.udp_port}")
    
    def _tcp_server(self):
        """TCP server for reliable data transmission"""
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.tcp_socket.bind(('0.0.0.0', self.tcp_port))
            self.tcp_socket.listen(5)
            print(f"TCP server listening on port {self.tcp_port}")
            
            while self.running:
                try:
                    client_socket, address = self.tcp_socket.accept()
                    print(f"TCP client connected: {address}")
                    
                    client_thread = threading.Thread(
                        target=self._handle_tcp_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error:
                    if self.running:
                        print("TCP server error")
                    break
                    
        except Exception as e:
            print(f"TCP server error: {e}")
        finally:
            if self.tcp_socket:
                self.tcp_socket.close()
    
    def _handle_tcp_client(self, client_socket, address):
        """Handle TCP client connection"""
        try:
            while self.running:
                data = client_socket.recv(1024)
                if not data:
                    break
                
                try:
                    # Parse JSON data
                    json_data = json.loads(data.decode('utf-8'))
                    self._process_sensor_data(json_data)
                    
                    # Send acknowledgment
                    client_socket.send(b"OK")
                    
                except json.JSONDecodeError:
                    client_socket.send(b"ERROR")
                    
        except Exception as e:
            print(f"TCP client error: {e}")
        finally:
            client_socket.close()
            print(f"TCP client disconnected: {address}")
    
    def _udp_server(self):
        """UDP server for high-frequency data"""
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        try:
            self.udp_socket.bind(('0.0.0.0', self.udp_port))
            print(f"UDP server listening on port {self.udp_port}")
            
            while self.running:
                try:
                    data, address = self.udp_socket.recvfrom(1024)
                    
                    try:
                        # Parse JSON data
                        json_data = json.loads(data.decode('utf-8'))
                        self._process_sensor_data(json_data)
                        
                    except json.JSONDecodeError:
                        continue
                        
                except socket.error:
                    if self.running:
                        print("UDP server error")
                    break
                    
        except Exception as e:
            print(f"UDP server error: {e}")
        finally:
            if self.udp_socket:
                self.udp_socket.close()
    
    def _process_sensor_data(self, data):
        """Process incoming sensor data"""
        try:
            # Expected JSON format:
            # {
            #     "timestamp": 1234567890.123,
            #     "accelerometer": {"x": 0.1, "y": 0.2, "z": 9.8},
            #     "gyroscope": {"x": 0.01, "y": 0.02, "z": 0.03},
            # }
            
            timestamp = data.get('timestamp', time.time())
            accel = data.get('accelerometer', {})
            gyro = data.get('gyroscope', {})
          
            
            imu_data = IMUData(
                timestamp=timestamp,
                accel_x=accel.get('x', 0.0),
                accel_y=accel.get('y', 0.0),
                accel_z=accel.get('z', 0.0),
                gyro_x=gyro.get('x', 0.0),
                gyro_y=gyro.get('y', 0.0),
                gyro_z=gyro.get('z', 0.0),
            )
            
            self.imu_queue.put(imu_data)
            
        except Exception as e:
            print(f"Error processing sensor data: {e}")
    
    def get_imu_data(self):
        """Get latest IMU data from queue"""
        imu_data_list = []
        while not self.imu_queue.empty():
            try:
                imu_data_list.append(self.imu_queue.get_nowait())
            except queue.Empty:
                break
        return imu_data_list
    
    def stop_servers(self):
        """Stop network servers"""
        self.running = False
        
        if self.tcp_socket:
            self.tcp_socket.close()
        if self.udp_socket:
            self.udp_socket.close()

class Enhanced3DGuidance:
    """Enhanced 3D guidance system with IMU and optical flow"""
    
    def __init__(self):
        self.imu_processor = IMUProcessor()
        self.optical_flow = OpticalFlowProcessor()
        
        # 3D awareness parameters
        self.spatial_awareness = {
            'forward': deque(maxlen=10),
            'left': deque(maxlen=10),
            'right': deque(maxlen=10),
            'above': deque(maxlen=10),
            'below': deque(maxlen=10)
        }
        
        # Motion prediction
        self.motion_predictor = deque(maxlen=20)
        
    def update_motion_state(self, imu_data_list, optical_flow_data):
        """Update 3D motion state with IMU and optical flow"""
        # Process IMU data
        for imu_data in imu_data_list:
            self.imu_processor.add_imu_data(imu_data)
        
        # Get ego motion from optical flow
        flow_velocity, flow_confidence = self.optical_flow.get_ego_motion()
        
        # Fuse IMU and optical flow data
        imu_velocity = self.imu_processor.motion_state.velocity
        imu_confidence = self.imu_processor.motion_state.confidence
        
        # Weighted fusion of velocities
        total_confidence = imu_confidence + flow_confidence
        if total_confidence > 0:
            fused_velocity = (
                imu_velocity * imu_confidence + flow_velocity * flow_confidence
            ) / total_confidence
        else:
            fused_velocity = np.zeros(3)
        
        # Update motion predictor
        motion_data = {
            'velocity': fused_velocity,
            'position': self.imu_processor.motion_state.position,
            'orientation': self.imu_processor.motion_state.orientation,
            'confidence': total_confidence,
            'timestamp': time.time()
        }
        self.motion_predictor.append(motion_data)
        
        return motion_data
    
    def predict_collision_risk(self, detections, motion_data):
        """Predict collision risk based on 3D motion"""
        risks = []
        
        for detection in detections:
            # Get object position and velocity
            obj_pos = np.array([detection['center'][0], detection['center'][1], detection['depth']])
            obj_velocity = np.array([detection['velocity'], 0, 0])  # Simplified
            
            # Get user motion
            user_pos = motion_data['position']
            user_velocity = motion_data['velocity']
            
            # Calculate relative position and velocity
            relative_pos = obj_pos - user_pos
            relative_vel = obj_velocity - user_velocity
            
            # Time to collision
            relative_speed = np.linalg.norm(relative_vel)
            if relative_speed > 0.1:
                time_to_collision = np.linalg.norm(relative_pos) / relative_speed
            else:
                time_to_collision = float('inf')
            
            # Risk assessment
            if time_to_collision < 2.0:  # Less than 2 seconds
                risk_level = "HIGH"
            elif time_to_collision < 5.0:  # Less than 5 seconds
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            risks.append({
                'detection': detection,
                'time_to_collision': time_to_collision,
                'risk_level': risk_level,
                'relative_velocity': relative_vel
            })
        
        return sorted(risks, key=lambda x: x['time_to_collision'])
    
    def generate_3d_guidance(self, risks, motion_data):
        """Generate 3D spatial guidance"""
        if not risks:
            return "Path clear, continue forward."
        
        guidance_parts = []
        high_risk_objects = [r for r in risks if r['risk_level'] == 'HIGH']
        
        if high_risk_objects:
            for risk in high_risk_objects[:2]:  # Top 2 high-risk objects
                obj = risk['detection']['label']
                pos = risk['detection']['h_pos']
                ttc = risk['time_to_collision']
                
                if ttc < 1.0:
                    guidance_parts.append(f"STOP! {obj} collision imminent {pos}")
                else:
                    guidance_parts.append(f"CAUTION: {obj} approaching {pos} ")
        
        # Provide directional guidance based on user motion
        user_velocity = motion_data['velocity']
        if np.linalg.norm(user_velocity) > 0.1:
            # User is moving, provide motion-aware guidance
            forward_speed = user_velocity[2]  # Assuming Z is forward
            if forward_speed > 0.5:
                guidance_parts.append("Moving forward, reduce speed")
            elif forward_speed < -0.5:
                guidance_parts.append("Moving backward")
        
        # Spatial awareness guidance
        medium_risk = [r for r in risks if r['risk_level'] == 'MEDIUM']
        if medium_risk:
            for risk in medium_risk[:1]:  # One medium risk object
                obj = risk['detection']['label']
                pos = risk['detection']['h_pos']
                guidance_parts.append(f"{obj} ahead {pos}")
        
        return ". ".join(guidance_parts[:3]) + "."

class EnhancedObstacleDetector:
    """Enhanced obstacle detector with IMU and optical flow integration"""
    
    def __init__(self, tcp_port=8888, udp_port=8889):
        # Initialize text-to-speech
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)
        
        # Load models
        self._load_models()
        
        # Initialize tracker
        self.tracker = DeepSort(max_age=50, n_init=3)
        
        # Network server for phone data
        self.network_server = NetworkServer(tcp_port, udp_port)
        
        # Enhanced 3D guidance system
        self.guidance_3d = Enhanced3DGuidance()
        
        # Tracking and smoothing
        self.track_history = {}
        self.depth_history = {}
        self.velocity_history = {}
        self.current_tracks = []
        self.track_lock = threading.Lock()
        
        # Frame processing parameters
        self.frame_count = 0
        self.last_guidance_time = 0
        self.guidance_interval = 1.0
        self.frames_per_guidance = 30
        self.last_process_time = time.time()
        
        # Performance optimizations
        self.skip_frames = 2
        self.depth_skip_frames = 5
        self.last_depth_map = None
        self.resize_factor = 0.5
        
        # Depth scaling and thresholds
        self.max_depth_meters = 1.0
        self.depth_scale_factor = 5.0
        self.guidance_range = 2.0
        self.alert_threshold = 0.8
        self.critical_speed = 15.0
        
        # Audio queue
        self.audio_queue = queue.Queue(maxsize=3)
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
        
        # Priority classes
        self.priority_classes = {
            'person': 1, 'bicycle': 1, 'car': 1, 'motorcycle': 1, 'bus': 1, 'truck': 1,
            'chair': 2, 'dining table': 2, 'door': 2, 'stairs': 1,
            'handbag': 3, 'backpack': 3, 'umbrella': 3, 'bottle': 3
        }
        
        # Video source
        self.video_source = 1  # Default webcam
    def get_valid_depth(self, depth_map, bbox):
        """Returns reliable depth estimate or None"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validate bounding box
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Get central region (more stable)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        patch_size = min(30, (x2 - x1) // 2, (y2 - y1) // 2)
        
        patch = depth_map[
            max(0, center_y-patch_size):min(depth_map.shape[0], center_y+patch_size),
            max(0, center_x-patch_size):min(depth_map.shape[1], center_x+patch_size)
        ]
        
        if patch.size == 0:
            return None
            
        # Remove outliers
        valid_depths = patch[(patch > 0.5) & (patch < self.max_display_distance)]
        if len(valid_depths) == 0:
            return None
            
        return np.median(valid_depths)
    def get_velocity(self, track_id, current_pos):
        """Returns realistic velocity estimate"""
        if track_id not in self.track_history:
            self.track_history[track_id] = current_pos
            return 0.0, "stationary"
            
        prev_pos = self.track_history[track_id]
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        
        # Pixel displacement (capped at reasonable value)
        displacement = min(math.sqrt(dx*dx + dy*dy), 50)
        
        # Convert to m/s (assuming ~30fps)
        velocity = displacement * 0.033  # Adjusted scaling factor
        
        # Simple direction
        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
            
        return min(velocity, 5.0), direction  # Cap at 5 m/s
    def _load_models(self):
        """Load all ML models"""
        self.model = YOLO("yolov8n.pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load MiDaS
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.model.device)
        self.midas_transforms = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.midas.eval()
        
        # Lightweight LLM
        model_name = "microsoft/DialoGPT-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation", 
            model=self.llm, 
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        self.names = self.model.names
    
    def estimate_depth(self, bbox, frame_size):
        """Basic depth estimation based on bounding box size"""
        # This is a placeholder - replace with actual depth estimation
        # Currently uses inverse of bbox area as proxy for distance
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        frame_area = frame_size[0] * frame_size[1]
        
        # Normalized area (0-1)
        norm_area = area / frame_area
        
        # Simple inverse relationship
        return min(10.0, 1.0 / (norm_area + 0.001))

    def _audio_worker(self):
        """Background thread for text-to-speech"""
        while True:
            try:
                text = self.audio_queue.get(timeout=1)
                if text:
                    self.tts.say(text)
                    self.tts.runAndWait()
            except queue.Empty:
                continue
    
    def speak(self, text):
        """Add text to speech queue"""
        try:
            self.audio_queue.put_nowait(text)
        except queue.Full:
            pass  # Skip if queue is full
    
    def get_enhanced_position(self, cx, cy, img_w, img_h, depth):
        """Enhanced position calculation"""
        r = cx / img_w
        if r < 0.15:
            h_pos = "far left"
        elif r < 0.25:
            h_pos = "left"
        elif r < 0.4:
            h_pos = "slight left"    
        elif r < 0.6:
            h_pos = "center"
        elif r < 0.75:
            h_pos = "slight right"
        elif r < 0.9:
            h_pos = "right"
        else: 
            h_pos = 'far right'    
        
        v_ratio = cy / img_h
        if v_ratio < 0.3:
            v_pos = "above"
        elif v_ratio > 0.7:
            v_pos = "below"
        else:
            v_pos = "level"
        
        if depth < 0.3:
            dist_cat = "very close"
        elif depth < 0.6:
            dist_cat = "close"
        elif depth < 1.0:
            dist_cat = "medium distance"
        else:
            dist_cat = "far"
        
        return h_pos, v_pos, dist_cat
    
    def get_smoothed_velocity(self, track_id, current_pos):
        """Calculate smoothed velocity"""
        if track_id not in self.velocity_history:
            self.velocity_history[track_id] = deque(maxlen=5)
        
        if track_id in self.track_history:
            prev_pos = self.track_history[track_id]
            dx = current_pos[0] - prev_pos[0]
            dy = current_pos[1] - prev_pos[1]
            
            # Use actual time delta
            current_time = time.time()
            dt = current_time - self.last_process_time
            self.last_process_time = current_time
            
            velocity = math.sqrt((dx/dt)**2 + (dy/dt)**2)
            self.velocity_history[track_id].append(velocity)
            smoothed_velocity = np.mean(self.velocity_history[track_id])
            
            if smoothed_velocity > 2:
                if abs(dx) > abs(dy):
                    direction = "moving right" if dx > 0 else "moving left"
                else:
                    direction = "approaching" if dy > 0 else "moving away"
            else:
                direction = "stationary"
            
            if smoothed_velocity > self.critical_speed:
                speed_level = "fast"
            elif smoothed_velocity > 8:
                speed_level = "moderate"
            else:
                speed_level = "slow"
        else:
            smoothed_velocity = 0
            direction = "stationary"
            speed_level = "slow"
        
        self.track_history[track_id] = current_pos
        return round(smoothed_velocity, 1), direction, speed_level
    
    def get_smoothed_depth(self, depth_map, box, orig_size, track_id):
        """Get smoothed depth estimation"""
        H, W = depth_map.shape
        x1, y1, x2, y2 = [int(b) for b in box]
        
        vx1 = max(0, min(W-1, int(x1 * W / orig_size[0])))
        vy1 = max(0, min(H-1, int(y1 * H / orig_size[1])))
        vx2 = max(0, min(W-1, int(x2 * W / orig_size[0])))
        vy2 = max(0, min(H-1, int(y2 * H / orig_size[1])))
        
        if vx2 > vx1 and vy2 > vy1:
            patch = depth_map[vy1:vy2, vx1:vx2]
            if patch.size > 0:
                depth_25 = np.percentile(patch, 25)
                depth_50 = np.percentile(patch, 50)
                depth_75 = np.percentile(patch, 75)
                current_depth = (depth_25 + depth_50 + depth_75) / 3
            else:
                current_depth = self.max_depth_meters
        else:
            current_depth = self.max_depth_meters
        
        if track_id not in self.depth_history:
            self.depth_history[track_id] = deque(maxlen=3)
        
        self.depth_history[track_id].append(current_depth)
        smoothed_depth = np.mean(self.depth_history[track_id])
        
        return max(0.05, smoothed_depth)
    
    def prioritize_detections(self, detections):
        """Prioritize detections based on relevance"""
        prioritized = []
        for detection in detections:
            label = detection.get('label', '')
            priority = self.priority_classes.get(label, 4)
            detection['priority'] = priority
            prioritized.append(detection)
        
        return sorted(prioritized, key=lambda x: (x['priority'], x['depth']))
    
    def process_frame(self, frame):
        # Run object detection
        results = self.model(frame, conf=self.detection_thresh, verbose=False)
        
        # Prepare detections for tracking
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls_id = int(box.cls)
                label = self.model.names[cls_id]
                
                detections.append(([x1, y1, x2-x1, y2-y1], conf, label))
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Estimate depth (placeholder - integrate actual depth estimation)
        frame_h, frame_w = frame.shape[:2]
        depth_map = np.ones((frame_h, frame_w)) * self.max_display_distance
        
        # Process each track
        current_detections = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Get depth estimate
            depth = self.estimate_depth((x1, y1, x2, y2), (frame_w, frame_h))
            
            # Get velocity
            velocity, direction = self.get_velocity(track_id, (cx, cy))
            
            # Store detection info
            current_detections.append({
                'track_id': track_id,
                'label': track.get_det_class(),
                'bbox': (x1, y1, x2, y2),
                'depth': depth,
                'velocity': velocity,
                'direction': direction
            })
            
            # Determine color based on distance
            if depth < 2.0:
                color = self.colors['close']
            elif depth < 5.0:
                color = self.colors['medium']
            else:
                color = self.colors['far']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with info
            label_text = f"{track.get_det_class()} {depth:.1f}m"
            if velocity > 0.5:  # Only show velocity if moving
                label_text += f" {velocity:.1f}m/s {direction}"
                
            cv2.putText(frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Update history
            self.track_history[track_id] = (cx, cy)
        
        self.frame_count += 1
        return frame, current_detections
    
    def _draw_motion_overlay(self, frame, motion_data, optical_flow_data):
        """Draw motion information overlay"""
        h, w = frame.shape[:2]
        
        # IMU status
        imu_status = f"IMU Conf: {motion_data['confidence']:.2f}"
        cv2.putText(frame, imu_status, (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # User velocity
        velocity_norm = np.linalg.norm(motion_data['velocity'])
        velocity_text = f"User Speed: {velocity_norm:.2f} m/s"
        cv2.putText(frame, velocity_text, (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Optical flow status
        if optical_flow_data:
            flow_conf = optical_flow_data.get('confidence', 0)
            flow_text = f"Flow Conf: {flow_conf:.2f}"
            cv2.putText(frame, flow_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Motion direction indicator (compass-like)
        center_x, center_y = w - 80, 80
        radius = 30
        
        # Draw compass circle
        cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Draw velocity vector
        if velocity_norm > 0.1:
            vel_x, vel_y = motion_data['velocity'][:2]
            arrow_length = min(radius, velocity_norm * 20)
            end_x = int(center_x + vel_x * arrow_length / velocity_norm)
            end_y = int(center_y - vel_y * arrow_length / velocity_norm)  # Flip Y for screen coords
            
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)
        
        # Cardinal directions
        cv2.putText(frame, "N", (center_x - 5, center_y - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "S", (center_x - 5, center_y + radius + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "E", (center_x + radius + 5, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "W", (center_x - radius - 15, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
         """Main execution loop with network server integration"""
         # Start network servers
         self.network_server.start_servers()
         
         cap = cv2.VideoCapture(self.video_source)
         if not cap.isOpened():
             print("Error: Could not open video source")
             return
         
         try:
             # Get video properties
             fps = cap.get(cv2.CAP_PROP_FPS)
             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
             original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
             original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
             
             print(f"Enhanced Obstacle Detection System Started")
             print(f"Video: {fps} FPS, {total_frames} frames, {original_width}x{original_height}")
             print(f"Network: TCP:{self.network_server.tcp_port}, UDP:{self.network_server.udp_port}")
             print(f"Features: IMU Integration, Optical Flow, 3D Motion Prediction")
             
             # Display settings
             display_width = 1280
             display_height = 720
             
             aspect_ratio = original_width / original_height
             if display_width / display_height > aspect_ratio:
                 display_width = int(display_height * aspect_ratio)
             else:
                 display_height = int(display_width / aspect_ratio)
             
             cv2.namedWindow("Enhanced AssistVision - IMU + Optical Flow", cv2.WINDOW_NORMAL)
             cv2.resizeWindow("Enhanced AssistVision - IMU + Optical Flow", display_width, display_height)
             
             # Adjust guidance frequency
             self.frames_per_guidance = max(10, int(fps))
             delay = max(1, int(1000 / fps))
             
             print("System ready. Connect phone sensors to:")
             print(f"  TCP: <your_ip>:{self.network_server.tcp_port}")
             print(f"  UDP: <your_ip>:{self.network_server.udp_port}")
             
             self.speak("Enhanced obstacle detection with IMU and optical flow started. Connect phone sensors for full 3D awareness.")
             
             start_time = time.time()
             while True:
                 ret, frame = cap.read()
                 if not ret:
                     print("End of video reached")
                     break
                 
                 # Process frame with enhanced features
                 processed_frame, detections, motion_data = self.process_frame(frame)
                 
                 # Performance info
                 elapsed = time.time() - start_time
                 current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                 
                 # Enhanced system info overlay
                 info_text = f"Frame: {self.frame_count} | FPS: {current_fps:.1f} | IMU: {len(self.network_server.get_imu_data())} samples"
                 cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                 
                 network_info = f"TCP/UDP Servers Active | 3D Motion Tracking | Collision Prediction"
                 cv2.putText(processed_frame, network_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                 
                 # Enhanced legend
                 legend_text = "Red:<30cm Orange:<60cm Yellow:<1m Green:>1m | ->=Fast =>=Moving"
                 cv2.putText(processed_frame, legend_text, 
                            (10, processed_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                 
                 control_text = "Controls: Q=Quit SPACE=Manual R=Reset F=FastMode T=Threshold I=IMU_Info"
                 cv2.putText(processed_frame, control_text, 
                            (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                 
                 # Display frame
                 display_frame = cv2.resize(processed_frame, (display_width, display_height))
                 cv2.imshow("Enhanced AssistVision - IMU + Optical Flow", display_frame)
                 
                 # Enhanced controls
                 key = cv2.waitKey(delay) & 0xFF
                 if key == ord('q'):
                     break
                 elif key == ord(' '):  # Manual guidance
                     if detections:
                         prioritized = self.prioritize_detections(detections)
                         collision_risks = self.guidance_3d.predict_collision_risk(prioritized, motion_data)
                         guidance = self.guidance_3d.generate_3d_guidance(collision_risks, motion_data)
                         print(f"Manual guidance: {guidance}")
                         self.speak(guidance)
                     else:
                         self.speak("No obstacles detected")
                 elif key == ord('r'):  # Reset video
                     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                     self.frame_count = 0
                     start_time = time.time()
                 elif key == ord('f'):  # Toggle fast mode
                     self.skip_frames = 0 if self.skip_frames > 0 else 2
                     print(f"Fast mode: {'ON' if self.skip_frames == 0 else 'OFF'}")
                 elif key == ord('t'):  # Adjust threshold
                     self.alert_threshold = 0.8 if self.alert_threshold > 1.0 else 1.2
                     print(f"Alert threshold: {self.alert_threshold}m")
                     self.speak(f"Alert threshold set to {self.alert_threshold} meters")
                 elif key == ord('i'):  # IMU info
                     imu_samples = len(self.network_server.get_imu_data())
                     motion_conf = motion_data['confidence']
                     info = f"IMU samples: {imu_samples}, Motion confidence: {motion_conf:.2f}"
                     print(info)
                     self.speak(f"IMU confidence {motion_conf:.1f}")
                 
                 self.frame_count += 1
                 
         except KeyboardInterrupt:
             print("Stopping...")
         finally:
             # Cleanup
             cap.release()
             cv2.destroyAllWindows()
             self.network_server.stop_servers()
             self.speak("Enhanced system stopped")

def create_phone_sensor_example():
    """Example code for phone sensor data transmission"""
    example_code = '''
    // Android/iOS phone sensor example (pseudo-code)
    // Use WebSocket, TCP socket, or UDP socket to send data
    
    import socket
    import json
    import time
    from sensors import accelerometer, gyroscope
    
    # TCP connection example
    def send_sensor_data_tcp():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('YOUR_COMPUTER_IP', 8888))  # Replace with actual IP
        
        while True:
            accel = accelerometer.get_reading()
            gyro = gyroscope.get_reading()
            
            data = {
                "timestamp": time.time(),
                "accelerometer": {"x": accel.x, "y": accel.y, "z": accel.z},
                "gyroscope": {"x": gyro.x, "y": gyro.y, "z": gyro.z},
            }
            
            sock.send(json.dumps(data).encode())
            response = sock.recv(1024)
            time.sleep(1/30)  # 30Hz update rate
    
    # UDP example (faster, for high-frequency data)
    def send_sensor_data_udp():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        while True:
            accel = accelerometer.get_reading()
            gyro = gyroscope.get_reading()
            
            data = {
                "timestamp": time.time(),
                "accelerometer": {"x": accel.x, "y": accel.y, "z": accel.z},
                "gyroscope": {"x": gyro.x, "y": gyro.y, "z": gyro.z}
            }
            
            sock.sendto(json.dumps(data).encode(), ('YOUR_COMPUTER_IP', 8889))
            time.sleep(1/100)  # 60Hz for high-frequency updates
    '''
    
    print("Phone Sensor Integration Example:")
    print(example_code)

if __name__ == "__main__":
    print("Enhanced Obstacle Detection System with IMU and Optical Flow")
    print("=" * 60)
    
    # Show phone integration example
    # create_phone_sensor_example()
    
    print("\nStarting enhanced detection system...")
    
    # Create and run detector
    detector = EnhancedObstacleDetector(tcp_port=8888, udp_port=8889)
    detector.run()
