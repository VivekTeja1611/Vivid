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
from typing import Tuple, Optional, List, Dict, Any
import logging
import warnings

# Suppress warnings and logs
warnings.filterwarnings("ignore")
LOGGER.setLevel("ERROR")
logging.getLogger("transformers").setLevel(logging.ERROR)

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

@dataclass
class DetectionResult:
    """Structure for detection results"""
    track_id: int
    label: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    depth: float
    velocity: float
    direction: str
    priority: int
    center: Tuple[int, int]
    h_pos: str
    v_pos: str
    dist_cat: str

class IMUProcessor:
    """Process IMU data for motion estimation - essential for blind navigation"""
    
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

class NetworkServer:
    """TCP and UDP server for receiving phone sensor data - critical for blind navigation"""
    
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
    """Enhanced 3D guidance system specifically designed for blind navigation"""
    
    def __init__(self):
        self.imu_processor = IMUProcessor()
        
        # 3D spatial awareness for navigation
        self.spatial_awareness = {
            'forward': deque(maxlen=10),
            'left': deque(maxlen=10),
            'right': deque(maxlen=10),
            'above': deque(maxlen=10),
            'below': deque(maxlen=10)
        }
        
        # Motion prediction for safety
        self.motion_predictor = deque(maxlen=20)
        
        # Navigation assistance
        self.path_history = deque(maxlen=50)
        self.obstacle_memory = {}
        
    def update_motion_state(self, imu_data_list):
        """Update 3D motion state with IMU data"""
        # Process IMU data
        for imu_data in imu_data_list:
            self.imu_processor.add_imu_data(imu_data)
        
        # Update motion predictor
        motion_data = {
            'velocity': self.imu_processor.motion_state.velocity,
            'position': self.imu_processor.motion_state.position,
            'orientation': self.imu_processor.motion_state.orientation,
            'confidence': self.imu_processor.motion_state.confidence,
            'timestamp': time.time()
        }
        self.motion_predictor.append(motion_data)
        
        return motion_data
    
    def predict_collision_risk(self, detections, motion_data):
        """Predict collision risk based on 3D motion - critical for blind safety"""
        risks = []
        
        for detection in detections:
            # Get object position and velocity
            obj_pos = np.array([detection.center[0], detection.center[1], detection.depth])
            obj_velocity = np.array([detection.velocity, 0, 0])  # Simplified
            
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
            
            # Risk assessment for blind navigation
            if time_to_collision < 1.5:  # Less than 1.5 seconds - immediate danger
                risk_level = "CRITICAL"
            elif time_to_collision < 3.0:  # Less than 3 seconds - high risk
                risk_level = "HIGH"
            elif time_to_collision < 5.0:  # Less than 5 seconds - moderate risk
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
    
    def generate_navigation_guidance(self, risks, motion_data):
        """Generate navigation guidance specifically for blind users"""
        if not risks:
            return "Path clear ahead. Safe to continue."
        
        guidance_parts = []
        critical_objects = [r for r in risks if r['risk_level'] == 'CRITICAL']
        high_risk_objects = [r for r in risks if r['risk_level'] == 'HIGH']
        
        # Critical immediate dangers
        if critical_objects:
            guidance_parts.append("STOP IMMEDIATELY!")
            for risk in critical_objects[:1]:  # Only announce the most critical
                obj = risk['detection'].label
                pos = risk['detection'].h_pos
                dist = risk['detection'].depth
                guidance_parts.append(f"{obj} directly ahead at {dist:.1f} meters")
        
        # High risk objects requiring caution
        elif high_risk_objects:
            for risk in high_risk_objects[:2]:  # Top 2 high-risk objects
                obj = risk['detection'].label
                pos = risk['detection'].h_pos
                dist = risk['detection'].depth
                
                if dist < 1.0:
                    guidance_parts.append(f"Caution: {obj} very close {pos}")
                else:
                    guidance_parts.append(f"{obj} approaching {pos} at {dist:.1f} meters")
        
        # General navigation guidance
        else:
            medium_risk = [r for r in risks if r['risk_level'] == 'MEDIUM']
            if medium_risk:
                for risk in medium_risk[:1]:  # One medium risk object
                    obj = risk['detection'].label
                    pos = risk['detection'].h_pos
                    dist = risk['detection'].depth
                    guidance_parts.append(f"{obj} detected {pos} at {dist:.1f} meters")
        
        # Add navigation suggestions based on user motion
        user_velocity = motion_data['velocity']
        if np.linalg.norm(user_velocity) > 0.2:
            # User is moving, provide motion-aware guidance
            if len(guidance_parts) > 0:
                guidance_parts.append("Recommend slowing down")
        
        return ". ".join(guidance_parts[:3]) + "."

class SafetyZoneManager:
    """Manages safety zones specifically for blind navigation"""
    
    def __init__(self):
        # Adjusted zones for blind users - more conservative
        self.critical_zone = 0.8   # meters - immediate stop
        self.danger_zone = 1.5     # meters - high alert
        self.warning_zone = 2.5    # meters - caution
        self.caution_zone = 4.0    # meters - awareness
        
        # Movement prediction with safety margins
        self.velocity_threshold = 0.3  # m/s
        self.approach_threshold = 3.0  # seconds to collision
        
    def get_zone_type(self, depth: float, velocity: float = 0.0) -> str:
        """Determine safety zone based on depth and velocity"""
        # Adjust zone based on approaching velocity with safety margin
        effective_depth = depth - (velocity * 3.0)  # 3-second prediction for safety
        
        if effective_depth <= self.critical_zone:
            return "CRITICAL"
        elif effective_depth <= self.danger_zone:
            return "DANGER"
        elif effective_depth <= self.warning_zone:
            return "WARNING"
        elif effective_depth <= self.caution_zone:
            return "CAUTION"
        else:
            return "SAFE"
    
    def predict_collision_time(self, depth: float, velocity: float) -> float:
        """Predict time to collision in seconds"""
        if velocity <= 0:
            return float('inf')
        return depth / velocity

class DepthEstimator:
    """Improved depth estimation using multiple methods"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_midas()
        
        # Calibration parameters
        self.focal_length = 800  # Estimated focal length in pixels
        self.baseline = 0.1  # Estimated baseline in meters for stereo
        self.reference_heights = {
            'person': 1.7,  # meters
            'car': 1.5,
            'bicycle': 1.0,
            'motorcycle': 1.2,
            'truck': 3.0,
            'bus': 3.0
        }
        
    def _load_midas(self):
        """Load MiDaS depth estimation model"""
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device)
            self.midas_transforms = Compose([
                Resize((256, 256)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.midas.eval()
            self.midas_available = True
        except Exception as e:
            print(f"Warning: Could not load MiDaS model: {e}")
            self.midas_available = False
    
    def estimate_depth_from_size(self, bbox: Tuple[int, int, int, int], 
                                label: str, frame_shape: Tuple[int, int]) -> float:
        """Estimate depth based on object size and known dimensions"""
        x1, y1, x2, y2 = bbox
        object_height_pixels = y2 - y1
        frame_height = frame_shape[0]
        
        # Get reference height for object type
        real_height = self.reference_heights.get(label, 1.0)
        
        # Simple perspective calculation
        if object_height_pixels > 0:
            # Depth = (real_height * focal_length) / pixel_height
            depth = (real_height * self.focal_length) / object_height_pixels
            
            # Normalize based on position in frame (objects lower in frame are closer)
            y_center = (y1 + y2) / 2
            position_factor = (frame_height - y_center) / frame_height
            depth *= (0.5 + position_factor)  # Adjust depth based on vertical position
            
            return max(0.5, min(20.0, depth))
        
        return 5.0  # Default depth
    
    def estimate_depth_midas(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Estimate depth using MiDaS model"""
        if not self.midas_available:
            return None
            
        try:
            # Prepare frame for MiDaS
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.midas_transforms(Image.fromarray(rgb_frame)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                depth_map = self.midas(input_tensor)
                depth_map = torch.nn.functional.interpolate(
                    depth_map.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Extract depth from bounding box region
            x1, y1, x2, y2 = bbox
            depth_region = depth_map[y1:y2, x1:x2]
            
            if depth_region.numel() > 0:
                # Use median depth from central region
                central_depth = torch.median(depth_region).item()
                
                # Convert MiDaS output to meters (approximate)
                depth_meters = 10.0 / (central_depth + 1e-6)
                return max(0.3, min(20.0, depth_meters))
                
        except Exception as e:
            print(f"MiDaS depth estimation error: {e}")
        
        return None
    
    def get_depth(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                  label: str) -> float:
        """Get best depth estimate using available methods"""
        # Try MiDaS first
        midas_depth = self.estimate_depth_midas(frame, bbox)
        
        # Get size-based estimate as fallback
        size_depth = self.estimate_depth_from_size(bbox, label, frame.shape)
        
        # Combine estimates if both available
        if midas_depth is not None:
            # Weight MiDaS more heavily but use size estimate for validation
            if abs(midas_depth - size_depth) < 5.0:  # Estimates agree
                return (midas_depth * 0.7 + size_depth * 0.3)
            else:
                return midas_depth  # Trust MiDaS when estimates differ significantly
        
        return size_depth

class VelocityTracker:
    """Tracks object velocities with smoothing"""
    
    def __init__(self, history_size: int = 5):
        self.position_history: Dict[int, deque] = {}
        self.velocity_history: Dict[int, deque] = {}
        self.last_time: Dict[int, float] = {}
        self.history_size = history_size
        
    def update(self, track_id: int, position: Tuple[float, float], 
               timestamp: float = None) -> Tuple[float, str]:
        """Update position and calculate velocity"""
        if timestamp is None:
            timestamp = time.time()
            
        if track_id not in self.position_history:
            self.position_history[track_id] = deque(maxlen=self.history_size)
            self.velocity_history[track_id] = deque(maxlen=self.history_size)
            self.last_time[track_id] = timestamp
            self.position_history[track_id].append(position)
            return 0.0, "stationary"
        
        # Calculate velocity
        prev_position = self.position_history[track_id][-1]
        dt = timestamp - self.last_time[track_id]
        
        if dt > 0:
            dx = position[0] - prev_position[0]
            dy = position[1] - prev_position[1]
            
            # Convert pixel velocity to approximate m/s
            pixel_velocity = math.sqrt(dx*dx + dy*dy) / dt
            velocity_ms = pixel_velocity * 0.01  # Approximate conversion factor
            
            # Determine direction for blind navigation
            if pixel_velocity < 5:  # Threshold for stationary
                direction = "stationary"
            else:
                if abs(dx) > abs(dy):
                    direction = "moving right" if dx > 0 else "moving left"
                else:
                    direction = "approaching" if dy > 0 else "moving away"
            
            self.velocity_history[track_id].append(velocity_ms)
            smoothed_velocity = np.mean(list(self.velocity_history[track_id]))
            
        else:
            smoothed_velocity = 0.0
            direction = "stationary"
        
        # Update history
        self.position_history[track_id].append(position)
        self.last_time[track_id] = timestamp
        
        return smoothed_velocity, direction
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove data for tracks that are no longer active"""
        all_track_ids = list(self.position_history.keys())
        for track_id in all_track_ids:
            if track_id not in active_track_ids:
                self.position_history.pop(track_id, None)
                self.velocity_history.pop(track_id, None)
                self.last_time.pop(track_id, None)

class AudioFeedback:
    """Handles text-to-speech feedback with priority queue - critical for blind users"""
    
    def __init__(self):
        self.tts = None
        self._initialize_tts()
        self.audio_queue = queue.PriorityQueue(maxsize=10)  # Larger queue for blind navigation
        self.is_speaking = False
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
        
    def _initialize_tts(self):
        """Initialize text-to-speech engine with settings optimized for blind users"""
        try:
            self.tts = pyttsx3.init()
            self.tts.setProperty('rate', 160)  # Slightly slower for clarity
            self.tts.setProperty('volume', 1.0)  # Maximum volume
            
            # Try to set a clear voice
            voices = self.tts.getProperty('voices')
            if voices:
                # Prefer female voice if available (often clearer)
                for voice in voices:
                    if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                        self.tts.setProperty('voice', voice.id)
                        break
                else:
                    self.tts.setProperty('voice', voices[0].id)
                
        except Exception as e:
            print(f"Warning: Could not initialize TTS: {e}")
            self.tts = None
    
    def _audio_worker(self):
        """Background worker for audio output"""
        while True:
            try:
                priority, timestamp, text = self.audio_queue.get(timeout=1)
                if self.tts and text:
                    self.is_speaking = True
                    self.tts.say(text)
                    self.tts.runAndWait()
                    self.is_speaking = False
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio worker error: {e}")
                self.is_speaking = False
    
    def speak(self, text: str, priority: int = 5):
        """Add text to speech queue with priority (lower number = higher priority)"""
        if not self.tts:
            return
            
        try:
            timestamp = time.time()
            self.audio_queue.put_nowait((priority, timestamp, text))
        except queue.Full:
            # If queue is full, prioritize critical safety messages
            if priority <= 1:  # Critical safety messages
                try:
                    # Remove lowest priority messages
                    temp_items = []
                    while not self.audio_queue.empty():
                        item = self.audio_queue.get_nowait()
                        if item[0] <= 2:  # Keep high priority items
                            temp_items.append(item)
                    
                    # Put back high priority items
                    for item in temp_items:
                        self.audio_queue.put_nowait(item)
                    
                    # Add new critical message
                    self.audio_queue.put_nowait((priority, timestamp, text))
                except queue.Empty:
                    pass

class GuidanceGenerator:
    """Generates intelligent guidance specifically designed for blind users"""
    
    def __init__(self):
        # Priority classes optimized for blind navigation
        self.priority_classes = {
            'person': 1, 'bicycle': 1, 'car': 1, 'motorcycle': 1, 
            'bus': 1, 'truck': 1, 'traffic light': 1, 'stop sign': 1,
            'stairs': 1, 'escalator': 1, 'door': 2, 'chair': 2,
            'dining table': 2, 'bench': 2, 'pole': 2,
            'handbag': 4, 'backpack': 4, 'umbrella': 4, 'bottle': 5
        }
        
        self.last_guidance_time = 0
        self.guidance_interval = 1.5  # More frequent guidance for blind users
        self.last_guidance = ""
        
    def get_priority(self, label: str) -> int:
        """Get priority for object class"""
        return self.priority_classes.get(label, 5)
    
    def get_position_description(self, bbox: Tuple[int, int, int, int], 
                                frame_shape: Tuple[int, int]) -> Tuple[str, str]:
        """Get human-readable position description optimized for blind navigation"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        frame_h, frame_w = frame_shape
        
        # More precise horizontal position for navigation
        h_ratio = cx / frame_w
        if h_ratio < 0.15:
            h_pos = "far left"
        elif h_ratio < 0.35:
            h_pos = "left"
        elif h_ratio < 0.45:
            h_pos = "slightly left"
        elif h_ratio < 0.55:
            h_pos = "directly ahead"
        elif h_ratio < 0.65:
            h_pos = "slightly right"
        elif h_ratio < 0.85:
            h_pos = "right"
        else:
            h_pos = "far right"
        
        # Vertical position
        v_ratio = cy / frame_h
        if v_ratio < 0.3:
            v_pos = "above"
        elif v_ratio > 0.7:
            v_pos = "below"
        else:
            v_pos = "level"
            
        return h_pos, v_pos
    
    def get_distance_category(self, depth: float) -> str:
        """Categorize distance for speech with blind-specific terms"""
        if depth < 0.8:
            return "immediate proximity"
        elif depth < 1.5:
            return "very close"
        elif depth < 2.5:
            return "close"
        elif depth < 4.0:
            return "nearby"
        else:
            return "distant"
    
    def generate_guidance(self, detections: List[DetectionResult], 
                         force: bool = False) -> Optional[str]:
        """Generate guidance message specifically for blind navigation"""
        current_time = time.time()
        
        if not force and (current_time - self.last_guidance_time) < self.guidance_interval:
            return None
        
        if not detections:
            guidance = "Path appears clear"
        else:
            # Sort by priority and distance
            priority_detections = sorted(detections, 
                                       key=lambda x: (x.priority, x.depth))
            
            # Focus on most important objects for navigation
            navigation_objects = []
            for det in priority_detections:
                if det.depth < 5.0:  # Only mention nearby objects
                    navigation_objects.append(det)
            
            if not navigation_objects:
                guidance = "Path appears clear"
            else:
                guidance_parts = []
                
                # Process most critical objects first
                for i, det in enumerate(navigation_objects[:3]):  # Top 3 most important
                    safety_zone = SafetyZoneManager().get_zone_type(det.depth, det.velocity)
                    
                    if safety_zone == "CRITICAL":
                        guidance_parts.append(f"STOP! {det.label} {det.h_pos}")
                        break  # Stop processing if critical danger detected
                    elif safety_zone == "DANGER":
                        guidance_parts.append(f"Danger: {det.label} {det.h_pos} at {det.depth:.1f} meters")
                    elif safety_zone == "WARNING":
                        if det.velocity > 0.5:
                            guidance_parts.append(f"Moving {det.label} {det.h_pos}")
                        else:
                            guidance_parts.append(f"{det.label} {det.h_pos} at {det.depth:.1f} meters")
                    elif i == 0:  # Only mention first caution object to avoid information overload
                        guidance_parts.append(f"{det.label} {det.h_pos}")
                
                guidance = ". ".join(guidance_parts[:2])  # Limit to prevent information overload
        
        # Avoid repeating same guidance unless forced
        if guidance != self.last_guidance or force:
            self.last_guidance = guidance
            self.last_guidance_time = current_time
            return guidance
        
        return None

class EnhancedObstacleDetector:
    """Main obstacle detection system enhanced for blind navigation with phone sensor integration"""
    
    def __init__(self, detection_thresh: float = 0.5, max_display_distance: float = 10.0, 
                 tcp_port: int = 8888, udp_port: int = 8889):
        # Core parameters
        self.detection_thresh = detection_thresh
        self.max_display_distance = max_display_distance
        
        # Network server for phone integration
        self.network_server = NetworkServer(tcp_port, udp_port)
        
        # Enhanced 3D guidance system
        self.guidance_3d = Enhanced3DGuidance()
        
        # Initialize components
        self._initialize_models()
        self._initialize_components()
        
        # Performance tracking
        self.frame_count = 0
        self.last_process_time = time.time()
        
        # Colors for visualization (still useful for sighted assistants)
        self.colors = {
            'critical': (0, 0, 128),     # Dark red
            'danger': (0, 0, 255),       # Red
            'warning': (0, 165, 255),    # Orange
            'caution': (0, 255, 255),    # Yellow
            'safe': (0, 255, 0),         # Green
            'text': (255, 255, 255)      # White
        }
        
        print("Enhanced Obstacle Detector for Blind Navigation initialized successfully")
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # YOLO model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = YOLO("yolov8n.pt").to(self.device)
            self.names = self.model.names
            
            print(f"YOLO model loaded on {self.device}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize system components"""
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.depth_estimator = DepthEstimator()
        self.velocity_tracker = VelocityTracker()
        self.safety_zone = SafetyZoneManager()
        self.audio_feedback = AudioFeedback()
        self.guidance_generator = GuidanceGenerator()
        
    def start_phone_integration(self):
        """Start phone sensor integration servers"""
        self.network_server.start_servers()
        print("Phone sensor integration started")
        self.audio_feedback.speak("Phone sensor integration ready", priority=3)
    
    def stop_phone_integration(self):
        """Stop phone sensor integration servers"""
        self.network_server.stop_servers()
        print("Phone sensor integration stopped")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Process a single frame and return annotated frame with detections"""
        start_time = time.time()
        
        # Get IMU data from phone
        imu_data_list = self.network_server.get_imu_data()
        
        # Update motion state with IMU data
        motion_data = self.guidance_3d.update_motion_state(imu_data_list)
        
        # Run YOLO detection
        results = self.model(frame, conf=self.detection_thresh, verbose=False)
        
        # Prepare detections for tracking
        raw_detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf.cpu().numpy())
                    cls_id = int(box.cls.cpu().numpy())
                    label = self.names[cls_id]
                    
                    raw_detections.append(([x1, y1, x2-x1, y2-y1], conf, label))
        
        # Update tracker
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        
        # Process tracks into structured detections
        structured_detections = []
        active_track_ids = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            active_track_ids.append(track_id)
            
            # Get bounding box
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            bbox = (x1, y1, x2, y2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Get label and confidence
            label = track.get_det_class() if hasattr(track, 'get_det_class') else "object"
            confidence = track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.5
            
            # Estimate depth
            depth = self.depth_estimator.get_depth(frame, bbox, label)
            
            # Calculate velocity
            velocity, direction = self.velocity_tracker.update(track_id, (cx, cy))
            
            # Get position descriptions
            h_pos, v_pos = self.guidance_generator.get_position_description(bbox, frame.shape)
            dist_cat = self.guidance_generator.get_distance_category(depth)
            
            # Create detection result
            detection = DetectionResult(
                track_id=track_id,
                label=label,
                bbox=bbox,
                confidence=confidence,
                depth=depth,
                velocity=velocity,
                direction=direction,
                priority=self.guidance_generator.get_priority(label),
                center=(cx, cy),
                h_pos=h_pos,
                v_pos=v_pos,
                dist_cat=dist_cat
            )
            
            structured_detections.append(detection)
        
        # Clean up old tracks
        self.velocity_tracker.cleanup_old_tracks(active_track_ids)
        
        # Predict collision risks using 3D guidance
        collision_risks = self.guidance_3d.predict_collision_risk(structured_detections, motion_data)
        
        # Generate navigation guidance
        if collision_risks:
            guidance = self.guidance_3d.generate_navigation_guidance(collision_risks, motion_data)
        else:
            guidance = self.guidance_generator.generate_guidance(structured_detections)
        
        if guidance:
            # Determine priority based on safety zones
            has_critical = any(self.safety_zone.get_zone_type(d.depth, d.velocity) == "CRITICAL" 
                             for d in structured_detections)
            has_danger = any(self.safety_zone.get_zone_type(d.depth, d.velocity) == "DANGER" 
                           for d in structured_detections)
            
            if has_critical:
                priority = 0  # Highest priority
            elif has_danger:
                priority = 1  # High priority
            else:
                priority = 3  # Normal priority
                
            self.audio_feedback.speak(guidance, priority)
        
        # Annotate frame (for sighted assistants or debugging)
        annotated_frame = self._annotate_frame(frame, structured_detections, motion_data)
        
        # Update performance metrics
        self.frame_count += 1
        
        return annotated_frame, structured_detections
    
    def _annotate_frame(self, frame: np.ndarray, 
                       detections: List[DetectionResult],
                       motion_data: Dict) -> np.ndarray:
        """Annotate frame with detection results and motion data"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Determine color based on safety zone
            zone_type = self.safety_zone.get_zone_type(det.depth, det.velocity)
            color = self.colors.get(zone_type.lower(), self.colors['safe'])
            
            # Draw bounding box
            thickness = 4 if zone_type in ["CRITICAL", "DANGER"] else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label_parts = [det.label]
            label_parts.append(f"{det.depth:.1f}m")
            
            if det.velocity > 0.3:
                label_parts.append(f"{det.velocity:.1f}m/s")
                label_parts.append(det.direction[:4])  # Abbreviate direction
            
            label_text = " ".join(label_parts)
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(annotated, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
            
            # Draw center point
            cv2.circle(annotated, det.center, 4, color, -1)
        
        # Add system info including IMU data
        self._draw_system_info(annotated, detections, motion_data)
        
        return annotated
    
    def _draw_system_info(self, frame: np.ndarray, 
                         detections: List[DetectionResult],
                         motion_data: Dict):
        """Draw system information including IMU data on frame"""
        h, w = frame.shape[:2]
        
        # Performance info
        fps = self.frame_count / (time.time() - self.last_process_time + 0.001)
        info_text = f"FPS: {fps:.1f} | Objects: {len(detections)} | Frame: {self.frame_count}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # IMU status
        imu_confidence = motion_data.get('confidence', 0.0)
        imu_text = f"IMU Confidence: {imu_confidence:.2f}"
        cv2.putText(frame, imu_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # User velocity
        user_velocity = motion_data.get('velocity', np.zeros(3))
        velocity_norm = np.linalg.norm(user_velocity)
        velocity_text = f"User Speed: {velocity_norm:.2f} m/s"
        cv2.putText(frame, velocity_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Zone legend
        legend_y = h - 120
        zone_info = [
            ("CRITICAL (<0.8m)", self.colors['critical']),
            ("DANGER (<1.5m)", self.colors['danger']),
            ("WARNING (<2.5m)", self.colors['warning']),
            ("CAUTION (<4m)", self.colors['caution']),
            ("SAFE (>4m)", self.colors['safe'])
        ]
        
        for i, (text, color) in enumerate(zone_info):
            x_pos = 10 + i * 140
            cv2.rectangle(frame, (x_pos, legend_y), (x_pos + 15, legend_y + 15), color, -1)
            cv2.putText(frame, text, (x_pos + 20, legend_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        # Audio feedback status
        if self.audio_feedback.is_speaking:
            cv2.putText(frame, "🔊 Speaking...", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Network status
        imu_queue_size = len(self.network_server.get_imu_data()) if hasattr(self.network_server, 'get_imu_data') else 0
        network_text = f"Phone Connected: {imu_queue_size > 0} | IMU Queue: {imu_queue_size}"
        cv2.putText(frame, network_text, (10, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Controls
        controls = "Controls: Q=Quit | SPACE=Manual Guidance | R=Reset | P=Phone Status"
        cv2.putText(frame, controls, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def manual_guidance(self, detections: List[DetectionResult]):
        """Trigger manual guidance announcement"""
        guidance = self.guidance_generator.generate_guidance(detections, force=True)
        if guidance:
            self.audio_feedback.speak(guidance, priority=1)
            print(f"Manual guidance: {guidance}")
        else:
            self.audio_feedback.speak("No objects detected in current view", priority=1)
            print("Manual guidance: No objects detected")
    
    def run(self, video_source: int = 0):
        """Run the detection system with video input and phone integration"""
        # Start phone sensor integration
        self.start_phone_integration()
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        # Set video properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Enhanced Obstacle Detection System for Blind Navigation Started")
        print(f"Video source: {video_source}")
        print(f"Device: {self.device}")
        print(f"Phone sensor integration: TCP:{self.network_server.tcp_port}, UDP:{self.network_server.udp_port}")
        print("Press 'q' to quit, SPACE for manual guidance, 'r' to reset, 'p' for phone status")
        
        self.audio_feedback.speak("Enhanced navigation system started. Connect your phone for full 3D awareness.", priority=2)
        
        cv2.namedWindow("Enhanced AssistVision for Blind Navigation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Enhanced AssistVision for Blind Navigation", 1280, 720)
        
        self.last_process_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                processed_frame, detections = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("Enhanced AssistVision for Blind Navigation", processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Manual guidance
                    self.manual_guidance(detections)
                elif key == ord('r'):  # Reset
                    self.frame_count = 0
                    self.last_process_time = time.time()
                    print("System reset")
                    self.audio_feedback.speak("System reset", priority=2)
                elif key == ord('p'):  # Phone status
                    imu_data = self.network_server.get_imu_data()
                    status = f"Phone connection: {'Active' if len(imu_data) > 0 else 'No data'}"
                    print(status)
                    self.audio_feedback.speak(status, priority=2)
                
        except KeyboardInterrupt:
            print("\nStopping detection system...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.stop_phone_integration()
            self.audio_feedback.speak("Navigation system stopped", priority=1)
            print("Detection system stopped")

# Utility functions for external use
def create_detector(detection_thresh: float = 0.5, 
                   max_display_distance: float = 10.0,
                   tcp_port: int = 8888,
                   udp_port: int = 8889) -> EnhancedObstacleDetector:
    """Factory function to create detector instance with phone integration"""
    return EnhancedObstacleDetector(detection_thresh, max_display_distance, tcp_port, udp_port)

def create_phone_sensor_example():
    """Example code for phone sensor data transmission"""
    example_code = '''
    // Android/iOS phone sensor example for blind navigation assistance
    // Use TCP socket for reliable data transmission
    
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
            time.sleep(1/100)  # 100Hz for high-frequency updates
    '''
    
    print("Phone Sensor Integration Example for Blind Navigation:")
    print(example_code)

if __name__ == "__main__":
    print("Enhanced Obstacle Detection System for Blind Navigation")
    print("=" * 60)
    
    # Show phone integration example
    create_phone_sensor_example()
    
    print("\nStarting enhanced detection system...")
    
    # Create and run detector with phone integration
    detector = EnhancedObstacleDetector(tcp_port=8888, udp_port=8889)
    detector.run()
