# System Architecture - VIVID Project

## 🏗️ High-Level Architecture

VIVID follows a **multi-pipeline sensor fusion architecture** that combines computer vision, inertial sensors, and intelligent decision-making for real-time assistive navigation.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Layer   │    │ Processing Layer │    │  Output Layer   │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Camera Feed   │───▶│ • Object Detect. │───▶│ • Audio Guide   │
│ • IMU Sensors   │    │ • Depth Estimate │    │ • Visual Feed   │
│ • User Motion   │    │ • Motion Track   │    │ • Risk Alerts   │
└─────────────────┘    │ • Sensor Fusion  │    └─────────────────┘
                       └──────────────────┘
```

## 🔄 Three-Pipeline Design

### Pipeline 1: Visual Processing
```
Camera Frame → YOLO v8 → Object Detection → Bounding Boxes
             ↓
          MiDaS → Depth Estimation → Distance Map
             ↓
        DeepSORT → Multi-Object Tracking → Persistent IDs
```

### Pipeline 2: Motion Analysis  
```
IMU Data → Accelerometer/Gyroscope → Motion State
        ↓
Optical Flow → Camera Motion → Ego-Motion Compensation
        ↓
Sensor Fusion → Combined Motion Estimate
```

### Pipeline 3: Decision Making
```
Visual Tracks + Motion Data → Risk Assessment → Priority Queue
                           ↓
              Spatial Reasoning → 3D Audio Guidance
                           ↓
              Speech Synthesis → User Feedback
```

## 🧩 Component Architecture

### Core Components

```
EnhancedObstacleDetector (Main Controller)
├── YOLOv8Detector
│   ├── Model Loading
│   ├── Object Detection
│   └── Confidence Filtering
├── DepthEstimator (MiDaS)
│   ├── Monocular Depth
│   ├── Depth Map Generation
│   └── Distance Calculation
├── DeepSORTTracker
│   ├── Feature Extraction
│   ├── Kalman Filtering
│   └── Hungarian Assignment
├── IMUProcessor
│   ├── Sensor Data Buffer
│   ├── Motion Estimation
│   └── Coordinate Transform
├── DataFusion
│   ├── Multi-Sensor Integration
│   ├── Risk Assessment
│   └── Decision Logic
└── GuidanceSystem
    ├── Spatial Audio
    ├── Text-to-Speech
    └── Priority Management
```

## 📊 Data Flow Diagram

```
┌─────────────┐
│ Camera (30  │──┐
│ FPS)        │  │
└─────────────┘  │
                 ▼
┌─────────────┐  ┌─────────────────┐
│ Phone IMU   │─▶│   Frame Sync    │
│ (100 Hz)    │  │   & Buffer      │
└─────────────┘  └─────────────────┘
                         │
                         ▼
                 ┌─────────────────┐
                 │ Object Detection│
                 │ (YOLO v8)       │
                 └─────────────────┘
                         │
                         ▼
                 ┌─────────────────┐    ┌─────────────────┐
                 │ Depth Estimation│    │ Object Tracking │
                 │ (MiDaS)         │    │ (DeepSORT)      │
                 └─────────────────┘    └─────────────────┘
                         │                       │
                         └───────┬───────────────┘
                                 ▼
                         ┌─────────────────┐
                         │  Data Fusion    │
                         │  & Risk Assess  │
                         └─────────────────┘
                                 │
                                 ▼
                         ┌─────────────────┐
                         │ Guidance System │
                         │ (Audio Output)  │
                         └─────────────────┘
```

## 🎯 Processing Pipelines Detail

### 1. Computer Vision Pipeline

**Input:** BGR Video Frame (640x480 @ 30 FPS)

**Processing Steps:**
1. **Frame Preprocessing**
   ```python
   # Resize for performance
   if resize_factor < 1.0:
       frame = cv2.resize(frame, None, fx=factor, fy=factor)
   ```

2. **Object Detection (YOLO v8)**
   ```python
   # Every 3rd frame (performance optimization)
   if frame_count % (skip_frames + 1) == 0:
       results = model(frame, conf=0.5, iou=0.6)
   ```

3. **Depth Estimation (MiDaS)**
   ```python
   # Every 6th frame (heavy computation)
   if frame_count % (depth_skip_frames + 1) == 0:
       depth_map = midas_model(input_tensor)
   ```

4. **Multi-Object Tracking (DeepSORT)**
   ```python
   # Every frame (maintain continuity)
   tracks = tracker.update(detections, frame)
   ```

**Output:** List of tracked objects with 3D positions

### 2. Sensor Fusion Pipeline

**Input:** IMU data stream (100+ Hz) + Visual tracks (30 Hz)

**Processing Steps:**
1. **IMU Data Buffering**
   ```python
   # Circular buffer for sensor history
   imu_history = deque(maxlen=100)
   ```

2. **Motion State Estimation**
   ```python
   # Integrate acceleration for velocity
   # Apply coordinate transformations
   motion_state = estimate_motion(imu_data_list)
   ```

3. **Ego-Motion Compensation**
   ```python
   # Distinguish user motion from object motion
   compensated_velocity = object_velocity - user_velocity
   ```

**Output:** Motion-compensated object velocities

### 3. Decision Making Pipeline

**Input:** Tracked objects + Motion data + User context

**Processing Steps:**
1. **Risk Assessment**
   ```python
   risk_score = calculate_risk(distance, velocity, object_class)
   ```

2. **Spatial Prioritization**
   ```python
   # Prioritize by distance and trajectory
   priority_queue = sort_by_collision_risk(objects)
   ```

3. **Guidance Generation**
   ```python
   # Generate contextual audio guidance
   guidance_text = generate_spatial_guidance(priority_objects)
   ```

**Output:** Audio guidance commands

## 🔧 Threading Architecture

### Multi-Threading Design

```
Main Thread (GUI)
├── VideoThread
│   ├── Frame Capture (30 FPS)
│   ├── Processing Pipeline
│   └── Display Updates
├── NetworkThread
│   ├── TCP Server (IMU Commands)
│   ├── UDP Server (Sensor Data)
│   └── Data Buffering
├── AudioThread
│   ├── TTS Queue Processing
│   ├── Speech Synthesis
│   └── Audio Output
└── ProcessingThread
    ├── Heavy Computations
    ├── Model Inference
    └── Background Tasks
```

### Thread Communication

```python
# Thread-safe communication using Qt signals
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detection_signal = pyqtSignal(list)
    
    def run(self):
        # Processing loop
        processed_frame, detections = self.detector.process_frame(frame)
        
        # Emit signals to main thread
        self.change_pixmap_signal.emit(processed_frame)
        self.detection_signal.emit(detections)
```

## 📡 Network Architecture

### Client-Server Model

```
┌─────────────────┐         ┌─────────────────┐
│   Smartphone    │  WiFi   │   Computer      │
│   (Client)      │◄───────►│   (Server)      │
├─────────────────┤         ├─────────────────┤
│ • Sensor Data   │  UDP    │ • Processing    │
│ • IMU Stream    │ 8889    │ • AI Models     │
│ • JSON Format   │         │ • GUI Display   │
└─────────────────┘  TCP    └─────────────────┘
                     8888
```

### Communication Protocols

**UDP (Sensor Data):**
- High frequency (100+ Hz)
- Low latency
- Packet loss acceptable
- JSON format

**TCP (Commands):**
- Reliable delivery
- Configuration changes
- Status updates
- Error handling

## 🧠 AI Model Architecture

### YOLO v8 Integration

```python
# Model loading and configuration
model = YOLO("yolov8n.pt").to(device)
model.conf = 0.5  # Confidence threshold
model.iou = 0.6   # NMS threshold
```

**Processing Flow:**
1. Input preprocessing (640x640 normalization)
2. Neural network inference
3. Non-maximum suppression
4. Coordinate transformation
5. Class filtering (80 COCO classes)

### MiDaS Depth Estimation

```python
# Transform pipeline
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Inference process
input_tensor = transform(rgb_image).unsqueeze(0)
with torch.no_grad():
    depth_map = model(input_tensor)
```

**Depth Processing:**
1. RGB → Normalized tensor
2. DPT (Dense Prediction Transformer)
3. Relative depth map
4. Bicubic interpolation
5. Distance estimation

### DeepSORT Tracking

```python
# Feature extraction + tracking
features = feature_extractor(object_crops)
tracks = kalman_filter.update(detections, features)
```

**Tracking Pipeline:**
1. Feature extraction (CNN)
2. Appearance matching
3. Kalman filter prediction
4. Hungarian assignment
5. Track management

## 🔄 State Management

### Application State

```python
@dataclass
class SystemState:
    # Processing state
    is_running: bool = False
    is_paused: bool = False
    current_frame: Optional[np.ndarray] = None
    
    # Detection state
    active_tracks: Dict[int, TrackInfo] = field(default_factory=dict)
    last_detections: List[Dict] = field(default_factory=list)
    
    # Sensor state
    imu_connected: bool = False
    last_imu_data: Optional[IMUData] = None
    motion_state: Optional[MotionState] = None
    
    # Configuration state
    voice_enabled: bool = True
    detection_threshold: float = 0.5
    processing_resolution: Tuple[int, int] = (640, 480)
```

### Track Management

```python
# Persistent object tracking across frames
track_states = {
    track_id: {
        'first_seen': timestamp,
        'last_seen': timestamp,
        'position_history': deque(maxlen=30),
        'velocity_history': deque(maxlen=10),
        'confidence_history': deque(maxlen=5)
    }
}
```

## ⚡ Performance Optimizations

### Computational Optimizations

1. **Frame Skipping Strategy**
   ```python
   # Different skip rates for different models
   YOLO_SKIP = 2      # Every 3rd frame
   MIDAS_SKIP = 5     # Every 6th frame
   TRACKING_SKIP = 0  # Every frame
   ```

2. **Resolution Scaling**
   ```python
   # Reduce resolution for speed
   RESIZE_FACTOR = 0.5  # Process at 50% resolution
   ```

3. **GPU Memory Management**
   ```python
   # Clear GPU cache periodically
   if frame_count % 100 == 0:
       torch.cuda.empty_cache()
   ```

### Memory Optimizations

```python
# Circular buffers for sensor data
imu_buffer = deque(maxlen=100)    # ~1 second of IMU data
depth_cache = deque(maxlen=10)    # Cache depth maps
track_history = deque(maxlen=30)  # 1 second of tracking
```

## 🔍 Error Handling Architecture

### Exception Hierarchy

```
VividException (Base)
├── CameraError
│   ├── CameraNotFoundError
│   ├── CameraPermissionError
│   └── CameraTimeoutError
├── ModelError
│   ├── ModelLoadError
│   ├── InferenceError
│   └── CudaOutOfMemoryError
├── NetworkError
│   ├── ServerStartError
│   ├── ConnectionTimeoutError
│   └── DataFormatError
└── ProcessingError
    ├── FrameProcessingError
    ├── TrackingError
    └── SensorFusionError
```

### Recovery Strategies

```python
# Graceful degradation approach
try:
    # Full pipeline with all features
    result = process_with_all_features(frame)
except CudaOutOfMemoryError:
    # Fallback to CPU processing
    result = process_with_cpu_only(frame)
except ModelLoadError:
    # Use basic OpenCV detection
    result = process_with_fallback_detection(frame)
```

## 📈 Scalability Design

### Modular Architecture

```
Core Module (vivid.py)
├── Detection Module
├── Tracking Module  
├── Depth Module
├── IMU Module
├── Fusion Module
└── Guidance Module
```

Each module can be:
- **Independently updated**
- **Swapped with alternatives**
- **Configured separately**
- **Tested in isolation**

### Extension Points

```python
# Plugin architecture for new models
class DetectorPlugin:
    def detect(self, frame: np.ndarray) -> List[Detection]:
        pass

# Register new detectors
detector_registry = {
    'yolo8': YOLO8Detector,
    'yolo11': YOLO11Detector,  # Future version
    'custom': CustomDetector
}
```

This architecture ensures VIVID remains maintainable, extensible, and performant while handling the complexity of real-time multi-sensor fusion.
