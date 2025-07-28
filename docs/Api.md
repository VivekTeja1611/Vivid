# API Reference - VIVID Project

## 🔌 Core Classes and Methods

### EnhancedObstacleDetector

Main detection class that coordinates all computer vision components.

```python
class EnhancedObstacleDetector:
    def __init__(self, tcp_port=8888, udp_port=8889, resize_factor=0.5)
```

#### Parameters
- `tcp_port` (int): TCP port for IMU data streaming (default: 8888)
- `udp_port` (int): UDP port for sensor data (default: 8889)  
- `resize_factor` (float): Frame resize factor for performance (default: 0.5)

#### Methods

##### `process_frame(frame)`
Process a single video frame through the detection pipeline.

```python
def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict], Dict]
```

**Parameters:**
- `frame` (np.ndarray): Input BGR video frame

**Returns:**
- `processed_frame` (np.ndarray): Frame with detection overlays
- `detections` (List[Dict]): List of detection objects
- `motion_data` (Dict): IMU and motion information

**Detection Object Structure:**
```python
{
    'track_id': int,           # Unique tracking ID
    'label': str,              # Object class name
    'bbox': Tuple[int, int, int, int],  # (x1, y1, x2, y2)
    'center': Tuple[int, int], # (cx, cy)
    'depth': float,            # Estimated distance in meters
    'velocity': float,         # Object velocity (compensated)
    'direction': str,          # Movement direction
    'confidence': float        # Detection confidence (0-1)
}
```

##### `run()`
Main execution loop for real-time processing.

```python
def run(self) -> None
```

Starts the main processing loop with video capture, detection, and guidance.

##### `speak(text)`
Add text to the speech synthesis queue.

```python
def speak(self, text: str) -> None
```

**Parameters:**
- `text` (str): Text to be spoken via TTS

##### `get_current_detections()`
Get the most recent detection results.

```python
def get_current_detections(self) -> List[Dict]
```

**Returns:**
- List of current detection objects

---

### NetworkServer

Handles smartphone sensor data streaming.

```python
class NetworkServer:
    def __init__(self, tcp_port=8888, udp_port=8889)
```

#### Methods

##### `start_servers()`
Start both TCP and UDP servers for sensor data.

```python
def start_servers(self) -> None
```

##### `get_latest_imu_data()`
Retrieve the most recent IMU sensor data.

```python
def get_latest_imu_data(self) -> Optional[IMUData]
```

**Returns:**
- `IMUData` object or None if no data available

---

### IMUProcessor

Processes inertial measurement unit data for motion estimation.

```python
class IMUProcessor:
    def __init__(self, buffer_size=100)
```

#### Methods

##### `add_imu_data(imu_data)`
Add new IMU measurement to the processing buffer.

```python
def add_imu_data(self, imu_data: IMUData) -> None
```

##### `get_motion_state()`
Get current estimated motion state.

```python
def get_motion_state(self) -> MotionState
```

**Returns:**
- `MotionState` object with position, velocity, and orientation

---

### DepthEstimator

Monocular depth estimation using MiDaS.

```python
class DepthEstimator:
    def __init__(self, model_type="DPT_Large")
```

#### Methods

##### `estimate(frame)`
Estimate depth map from a single RGB frame.

```python
def estimate(self, frame: np.ndarray) -> np.ndarray
```

**Parameters:**
- `frame` (np.ndarray): Input RGB frame

**Returns:**
- Normalized depth map (np.ndarray)

---

## 📊 Data Structures

### IMUData
```python
@dataclass
class IMUData:
    timestamp: float
    accel_x: float      # Acceleration X (m/s²)
    accel_y: float      # Acceleration Y (m/s²) 
    accel_z: float      # Acceleration Z (m/s²)
    gyro_x: float       # Angular velocity X (rad/s)
    gyro_y: float       # Angular velocity Y (rad/s)
    gyro_z: float       # Angular velocity Z (rad/s)
    mag_x: float = 0.0  # Magnetometer X (optional)
    mag_y: float = 0.0  # Magnetometer Y (optional)
    mag_z: float = 0.0  # Magnetometer Z (optional)
```

### MotionState
```python
@dataclass  
class MotionState:
    position: np.ndarray         # 3D position estimate
    velocity: np.ndarray         # 3D velocity vector
    acceleration: np.ndarray     # 3D acceleration
    orientation: np.ndarray      # Quaternion [w, x, y, z]
    angular_velocity: np.ndarray # 3D angular velocity
    confidence: float            # Estimation confidence (0-1)
```

### DetectionInfo
```python
@dataclass
class DetectionInfo:
    track_id: int
    label: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]          # cx, cy
    depth: float                     # Estimated distance (meters)
    velocity: float                  # Object velocity
    direction: str                   # Movement direction descriptor
    speed_level: str                 # "fast", "moderate", "slow"
    h_pos: str                       # "left", "center", "right"
    v_pos: str                       # "above", "level", "below"
    dist_cat: str                    # "very close", "close", "medium", "far"
```

---

## 🎯 Usage Examples

### Basic Detection

```python
from vivid import EnhancedObstacleDetector
import cv2

# Initialize detector
detector = EnhancedObstacleDetector()

# Process single frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    processed_frame, detections, motion_data = detector.process_frame(frame)
    
    # Display results
    for detection in detections:
        print(f"Object: {detection['label']}")
        print(f"Distance: {detection['depth']:.2f}m")
        print(f"Position: {detection['h_pos']}")
```

### Real-time Processing

```python
# Start real-time detection
detector = EnhancedObstacleDetector()
detector.run()  # Runs until interrupted
```

### Custom Configuration

```python
# Custom network ports and processing settings
detector = EnhancedObstacleDetector(
    tcp_port=9999,
    udp_port=9998, 
    resize_factor=0.3  # Smaller frames for better performance
)

# Enable specific features
detector.enable_voice_guidance = True
detector.depth_skip_frames = 5  # Process depth every 5th frame
```

### IMU Data Integration

```python
from vivid import NetworkServer, IMUProcessor

# Setup sensor streaming
server = NetworkServer(tcp_port=8888, udp_port=8889)
server.start_servers()

# Process IMU data
imu_processor = IMUProcessor()

while True:
    imu_data = server.get_latest_imu_data()
    if imu_data:
        imu_processor.add_imu_data(imu_data)
        motion_state = imu_processor.get_motion_state()
        print(f"User velocity: {np.linalg.norm(motion_state.velocity):.2f} m/s")
```

---

## 🔧 Configuration Options

### Performance Tuning

```python
# Adjust these parameters in your code for different performance/accuracy tradeoffs

# Frame processing frequency
SKIP_FRAMES = 2          # Process every 3rd frame for YOLO
DEPTH_SKIP_FRAMES = 5    # Process every 6th frame for depth

# Input resolution
RESIZE_FACTOR = 0.5      # Resize frames to 50% for speed

# Detection thresholds  
CONFIDENCE_THRESHOLD = 0.5  # YOLO confidence threshold
IOU_THRESHOLD = 0.6         # Non-max suppression threshold

# Tracking parameters
MAX_DISAPPEARED = 50     # Frames before removing lost tracks
MIN_HITS = 3            # Minimum detections before confirming track
```

### Audio Guidance Settings

```python
# Guidance frequency and content
GUIDANCE_INTERVAL = 90   # Frames between guidance messages
MAX_OBJECTS_TO_ANNOUNCE = 3  # Limit announcements to most important objects

# Distance categories (in meters)
DISTANCE_CATEGORIES = {
    "very close": (0, 0.5),
    "close": (0.5, 1.5), 
    "medium": (1.5, 3.0),
    "far": (3.0, float('inf'))
}
```

---

## 🚨 Error Handling

### Common Exceptions

```python
try:
    detector = EnhancedObstacleDetector()
    detector.run()
except CameraNotFoundError:
    print("Camera not accessible. Check camera index or connection.")
except ModelLoadError:
    print("Failed to load AI models. Run download_models.py first.")
except NetworkError:
    print("Failed to start network servers. Check port availability.")
except CudaOutOfMemoryError:
    print("GPU memory exhausted. Try reducing resolution or use CPU mode.")
```

### Debug Mode

```python
# Enable debug output
detector = EnhancedObstacleDetector()
detector.debug_mode = True  # Prints processing times and statistics
detector.run()
```

---

## 📡 Network Protocol

### IMU Data Format (UDP)

```json
{
    "timestamp": 1642345678.123,
    "accelerometer": {
        "x": 0.123,
        "y": 0.456, 
        "z": 9.789
    },
    "gyroscope": {
        "x": 0.012,
        "y": 0.034,
        "z": 0.056
    },
    "magnetometer": {
        "x": 45.67,
        "y": -12.34,
        "z": 23.45
    }
}
```

### Server Response Format (TCP)

```json
{
    "status": "success",
    "detections_count": 3,
    "timestamp": 1642345678.123,
    "motion_confidence": 0.85
}
```
