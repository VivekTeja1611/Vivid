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

class SafetyZoneManager:
    """Manages safety zones and collision prediction"""
    
    def __init__(self):
        self.danger_zone = 1.0  # meters
        self.warning_zone = 2.0  # meters
        self.caution_zone = 3.0  # meters
        
        # Movement prediction
        self.velocity_threshold = 0.5  # m/s
        self.approach_threshold = 2.0  # seconds to collision
        
    def get_zone_type(self, depth: float, velocity: float = 0.0) -> str:
        """Determine safety zone based on depth and velocity"""
        # Adjust zone based on approaching velocity
        effective_depth = depth - (velocity * 2.0)  # 2-second prediction
        
        if effective_depth <= self.danger_zone:
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
            
            # Determine direction
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
    """Handles text-to-speech feedback with priority queue"""
    
    def __init__(self):
        self.tts = None
        self._initialize_tts()
        self.audio_queue = queue.PriorityQueue(maxsize=5)
        self.is_speaking = False
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
        
    def _initialize_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts = pyttsx3.init()
            self.tts.setProperty('rate', 180)
            self.tts.setProperty('volume', 0.9)
            
            # Try to set a clear voice
            voices = self.tts.getProperty('voices')
            if voices:
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
            # If queue is full, ignore non-critical messages
            if priority <= 2:  # Only keep high priority messages
                try:
                    self.audio_queue.get_nowait()  # Remove oldest
                    self.audio_queue.put_nowait((priority, timestamp, text))
                except queue.Empty:
                    pass

class GuidanceGenerator:
    """Generates intelligent guidance based on detections"""
    
    def __init__(self):
        self.priority_classes = {
            'person': 1, 'bicycle': 1, 'car': 1, 'motorcycle': 1, 
            'bus': 1, 'truck': 1, 'traffic light': 2,
            'chair': 3, 'dining table': 3, 'door': 2, 
            'handbag': 4, 'backpack': 4, 'umbrella': 4, 'bottle': 4
        }
        
        self.last_guidance_time = 0
        self.guidance_interval = 2.0  # seconds
        self.last_guidance = ""
        
    def get_priority(self, label: str) -> int:
        """Get priority for object class"""
        return self.priority_classes.get(label, 5)
    
    def get_position_description(self, bbox: Tuple[int, int, int, int], 
                                frame_shape: Tuple[int, int]) -> Tuple[str, str]:
        """Get human-readable position description"""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        frame_h, frame_w = frame_shape
        
        # Horizontal position
        h_ratio = cx / frame_w
        if h_ratio < 0.2:
            h_pos = "far left"
        elif h_ratio < 0.4:
            h_pos = "left"
        elif h_ratio < 0.6:
            h_pos = "center"
        elif h_ratio < 0.8:
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
        """Categorize distance for speech"""
        if depth < 1.0:
            return "very close"
        elif depth < 2.0:
            return "close"
        elif depth < 4.0:
            return "nearby"
        else:
            return "distant"
    
    def generate_guidance(self, detections: List[DetectionResult], 
                         force: bool = False) -> Optional[str]:
        """Generate guidance message based on detections"""
        current_time = time.time()
        
        if not force and (current_time - self.last_guidance_time) < self.guidance_interval:
            return None
        
        if not detections:
            guidance = "Path clear"
        else:
            # Sort by priority and distance
            priority_detections = sorted(detections, 
                                       key=lambda x: (x.priority, x.depth))
            
            # Focus on most important objects
            important_objects = []
            for det in priority_detections[:3]:  # Top 3 most important
                if det.depth < 4.0:  # Only mention nearby objects
                    important_objects.append(det)
            
            if not important_objects:
                guidance = "Path clear"
            else:
                guidance_parts = []
                
                for det in important_objects:
                    safety_zone = SafetyZoneManager().get_zone_type(det.depth, det.velocity)
                    
                    if safety_zone == "DANGER":
                        guidance_parts.append(f"STOP! {det.label} {det.h_pos}")
                    elif safety_zone == "WARNING":
                        guidance_parts.append(f"Caution, {det.label} {det.h_pos}")
                    elif det.velocity > 0.5:
                        guidance_parts.append(f"{det.label} {det.direction} {det.h_pos}")
                    else:
                        guidance_parts.append(f"{det.label} {det.h_pos}")
                
                guidance = ". ".join(guidance_parts[:2])  # Limit to 2 objects
        
        # Avoid repeating same guidance
        if guidance != self.last_guidance or force:
            self.last_guidance = guidance
            self.last_guidance_time = current_time
            return guidance
        
        return None

class EnhancedObstacleDetector:
    """Main obstacle detection system with improved architecture"""
    
    def __init__(self, detection_thresh: float = 0.5, max_display_distance: float = 10.0):
        # Core parameters
        self.detection_thresh = detection_thresh
        self.max_display_distance = max_display_distance
        
        # Initialize components
        self._initialize_models()
        self._initialize_components()
        
        # Performance tracking
        self.frame_count = 0
        self.last_process_time = time.time()
        
        # Colors for visualization
        self.colors = {
            'danger': (0, 0, 255),    # Red
            'warning': (0, 165, 255), # Orange
            'caution': (0, 255, 255), # Yellow
            'safe': (0, 255, 0),      # Green
            'text': (255, 255, 255)   # White
        }
        
        print("Enhanced Obstacle Detector initialized successfully")
    
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
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Process a single frame and return annotated frame with detections"""
        start_time = time.time()
        
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
        
        # Generate guidance
        guidance = self.guidance_generator.generate_guidance(structured_detections)
        if guidance:
            # Determine priority based on safety zones
            has_danger = any(self.safety_zone.get_zone_type(d.depth, d.velocity) == "DANGER" 
                           for d in structured_detections)
            priority = 1 if has_danger else 3
            self.audio_feedback.speak(guidance, priority)
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame, structured_detections)
        
        # Update performance metrics
        self.frame_count += 1
        process_time = time.time() - start_time
        
        return annotated_frame, structured_detections
    
    def _annotate_frame(self, frame: np.ndarray, 
                       detections: List[DetectionResult]) -> np.ndarray:
        """Annotate frame with detection results"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Determine color based on safety zone
            zone_type = self.safety_zone.get_zone_type(det.depth, det.velocity)
            color = self.colors.get(zone_type.lower(), self.colors['safe'])
            
            # Draw bounding box
            thickness = 3 if zone_type in ["DANGER", "WARNING"] else 2
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
        
        # Add system info
        self._draw_system_info(annotated, detections)
        
        return annotated
    
    def _draw_system_info(self, frame: np.ndarray, 
                         detections: List[DetectionResult]):
        """Draw system information on frame"""
        h, w = frame.shape[:2]
        
        # Performance info
        fps = self.frame_count / (time.time() - self.last_process_time + 0.001)
        info_text = f"FPS: {fps:.1f} | Objects: {len(detections)} | Frame: {self.frame_count}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Zone legend
        legend_y = h - 80
        zone_info = [
            ("DANGER (<1m)", self.colors['danger']),
            ("WARNING (<2m)", self.colors['warning']),
            ("CAUTION (<3m)", self.colors['caution']),
            ("SAFE (>3m)", self.colors['safe'])
        ]
        
        for i, (text, color) in enumerate(zone_info):
            x_pos = 10 + i * 150
            cv2.rectangle(frame, (x_pos, legend_y), (x_pos + 15, legend_y + 15), color, -1)
            cv2.putText(frame, text, (x_pos + 20, legend_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        # Audio feedback status
        if self.audio_feedback.is_speaking:
            cv2.putText(frame, "🔊 Speaking...", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Controls
        controls = "Controls: Q=Quit | SPACE=Manual Guidance | R=Reset"
        cv2.putText(frame, controls, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def manual_guidance(self, detections: List[DetectionResult]):
        """Trigger manual guidance announcement"""
        guidance = self.guidance_generator.generate_guidance(detections, force=True)
        if guidance:
            self.audio_feedback.speak(guidance, priority=1)
            print(f"Manual guidance: {guidance}")
        else:
            self.audio_feedback.speak("No objects detected", priority=1)
            print("Manual guidance: No objects detected")
    
    def run(self, video_source: int = 0):
        """Run the detection system with video input"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        # Set video properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Enhanced Obstacle Detection System Started")
        print(f"Video source: {video_source}")
        print(f"Device: {self.device}")
        print("Press 'q' to quit, SPACE for manual guidance, 'r' to reset")
        
        self.audio_feedback.speak("Enhanced obstacle detection system started", priority=2)
        
        cv2.namedWindow("Enhanced AssistVision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Enhanced AssistVision", 1280, 720)
        
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
                cv2.imshow("Enhanced AssistVision", processed_frame)
                
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
                
        except KeyboardInterrupt:
            print("\nStopping detection system...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.audio_feedback.speak("Detection system stopped", priority=1)
            print("Detection system stopped")

# Utility functions for external use
def create_detector(detection_thresh: float = 0.5, 
                   max_display_distance: float = 10.0) -> EnhancedObstacleDetector:
    """Factory function to create detector instance"""
    return EnhancedObstacleDetector(detection_thresh, max_display_distance)

if __name__ == "__main__":
    print("Enhanced Obstacle Detection System")
    print("=" * 50)
    
    # Create and run detector
    detector = EnhancedObstacleDetector()
    detector.run()
