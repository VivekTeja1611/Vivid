# VIVID: Visually Impaired Vision-Integrated Device

[![Demo Video](https://img.shields.io/badge/Demo-Video-red?style=for-the-badge)](https://vivekteja1611.github.io/Vivid-presentation/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge)](https://github.com/vivekteja1611/Vivid)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://python.org)

## 🚀 Overview

VIVID is a **real-time assistive navigation system** that helps visually impaired individuals navigate their environment safely using computer vision and sensor fusion. The system combines smartphone sensors with advanced AI models to provide intelligent audio guidance for obstacle avoidance and navigation.

### ✨ Key Features

- **Real-time Object Detection** - Identifies obstacles, people, and hazards using YOLO v8
- **Depth Perception** - Estimates distances using monocular depth estimation (MiDaS)
- **Multi-Object Tracking** - Maintains awareness of moving objects with DeepSORT
- **Smart Audio Guidance** - Provides contextual voice feedback for navigation
- **Sensor Fusion** - Integrates smartphone IMU data for enhanced motion awareness
- **Cross-Platform** - Works on Windows, macOS, and Linux

## 🎯 Problem Statement

According to WHO, approximately **285 million people worldwide are visually impaired**. Current navigation aids like white canes and guide dogs have limitations:

- Limited detection range (1-2 meters)
- Cannot identify overhead obstacles
- No object classification capability
- Cannot predict moving object trajectories

**Our Solution:** A smartphone-based system combining computer vision with sensor data for comprehensive spatial awareness.

## 🏗️ System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Camera Feed │───▶│ YOLO v8      │───▶│                 │
└─────────────┘    │ Object       │    │                 │
                   │ Detection    │    │   Data Fusion   │───▶ Audio
┌─────────────┐    └──────────────┘    │   & Risk        │     Guidance
│ Phone       │───▶┌──────────────┐───▶│   Assessment    │
│ Sensors     │    │ IMU & Motion │    │                 │
└─────────────┘    │ Processing   │    │                 │
                   └──────────────┘    └─────────────────┘
┌─────────────┐    ┌──────────────┐           ▲
│ Depth       │───▶│ MiDaS Depth  │───────────┘
│ Estimation  │    │ & DeepSORT   │
└─────────────┘    │ Tracking     │
                   └──────────────┘
```

## 🛠️ Technologies Used

### Core Computer Vision
- **YOLO v8** - Real-time object detection (45-60 FPS)
- **MiDaS** - Monocular depth estimation
- **DeepSORT** - Multi-object tracking with persistent IDs
- **OpenCV** - Image processing and computer vision operations

### Sensor Integration
- **IMU Processing** - Smartphone accelerometer/gyroscope data
- **Optical Flow** - Camera motion estimation
- **TCP/UDP Networking** - Real-time sensor data streaming

### User Interface
- **PyQt5** - Real-time GUI with video display
- **Text-to-Speech** - Audio guidance system
- **Multi-threading** - Smooth performance and responsiveness

## 📋 Requirements

### System Requirements
- **OS:** Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **Python:** 3.8 or higher
- **RAM:** Minimum 8GB, recommended 16GB
- **GPU:** NVIDIA GPU with CUDA support (recommended but optional)
- **Storage:** 5GB free space

### Hardware Requirements
- **Camera:** Webcam or smartphone camera (via DroidCam)
- **Smartphone:** Android device with sensor capabilities
- **Network:** WiFi connection for sensor streaming

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/vivekteja1611/Vivid.git
cd Vivid

# Create virtual environment
python -m venv vivid_env
source vivid_env/bin/activate  # On Windows: vivid_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python download_models.py
```

### 2. Basic Usage

```bash
# Run with default camera (camera 0)
python app.py

# Run with DroidCam (usually camera 1)
python app.py --source 1

# Run with specific camera URL
python app.py --source http://192.168.1.100:4747/video
```

### 3. Smartphone Sensor Integration

#### Setup DroidCam
1. Install DroidCam on both mobile and computer
2. Connect via USB or WiFi
3. Start DroidCam and note the camera source

#### Setup Sensor Streaming (Android)
1. Install Termux from F-Droid
2. Setup sensor streaming:

```bash
# In Termux
pkg update && pkg upgrade
pkg install python python-numpy
pip install requests

# Create sensor streaming script
# (See detailed setup guide below)
```

## 📱 Detailed Setup Guide

### Mobile Device Preparation

1. **Install Termux** (from F-Droid, not Play Store)
2. **Update and install Python:**
   ```bash
   pkg update && pkg upgrade
   pkg install python python-numpy openssh
   pip install requests pyzmq
   ```

3. **Create sensor streaming script** (`sensor_stream.py`):
   ```python
   import subprocess, socket, json, time
   
   LAPTOP_IP = "YOUR_LAPTOP_IP"  # Replace with your laptop's IP
   PORT = 8889
   
   def get_sensor_data():
       proc = subprocess.Popen(
           ["termux-sensor", "-s", "accelerometer", "gyroscope"],
           stdout=subprocess.PIPE, text=True
       )
       
       buffer = ""
       for line in proc.stdout:
           buffer += line.strip()
           try:
               data = json.loads(buffer)
               yield json.dumps(data).encode()
               buffer = ""
           except json.JSONDecodeError:
               continue
   
   sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   print("Sending sensor data...")
   
   for data in get_sensor_data():
       sock.sendto(data, (LAPTOP_IP, PORT))
   ```

4. **Run the sensor script:**
   ```bash
   python sensor_stream.py
   ```

### Desktop Application Setup

1. **Find your IP address:**
   ```bash
   # Windows
   ipconfig
   
   # Linux/Mac
   ifconfig
   ```

2. **Start the VIVID application:**
   ```bash
   python app.py --source 1  # For DroidCam
   ```

## 🎮 Usage Instructions

### Main Interface
- **Video Display:** Shows real-time camera feed with object detection
- **Start/Pause:** Control video processing
- **Voice Guidance:** Toggle audio feedback
- **Settings:** Adjust detection parameters

### Voice Guidance Examples
- *"Person approaching center, 1.2 meters, moving left"*
- *"Chair detected right side, close distance"*
- *"Path clear ahead, safe to proceed"*

### Keyboard Shortcuts
- `Space` - Start/Pause processing
- `V` - Toggle voice guidance
- `S` - Save current frame
- `Q` - Quit application

## ⚙️ Configuration

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | Video source (0, 1, URL) | 0 |
| `--tcp_port` | IMU TCP port | 8888 |
| `--udp_port` | IMU UDP port | 8889 |
| `--no_imu` | Disable IMU integration | False |
| `--resolution` | Camera resolution | 640x480 |
| `--fps` | Target frame rate | 30 |

### Performance Tuning

For better performance on slower systems:

```bash
# Reduce processing load
python app.py --source 0 --resolution 320x240 --fps 15

# CPU-only mode (if GPU issues)
export CUDA_VISIBLE_DEVICES=""
python app.py
```

## 📊 Performance Metrics

| Component | Target | Achieved | Notes |
|-----------|---------|----------|--------|
| Overall FPS | 30 | 25-30 | Varies with scene complexity |
| YOLO Inference | <30ms | 25-35ms | GPU dependent |
| Depth Estimation | <50ms | 40-60ms | Largest bottleneck |
| Object Tracking | <20ms | 15-25ms | Scales with object count |
| Memory Usage | <2GB | 1.5-2.2GB | Peak during initialization |

## 🐛 Troubleshooting

### Common Issues

**Low Frame Rate:**
- Reduce input resolution: `--resolution 320x240`
- Enable frame skipping in code
- Use GPU acceleration if available

**IMU Connection Failed:**
- Check firewall settings
- Ensure devices on same WiFi network
- Verify IP address configuration

**High Memory Usage:**
- Use smaller YOLO model (yolov8n instead of yolov8l)
- Reduce batch processing size
- Close other applications

**Camera Not Detected:**
- Try different camera indices (0, 1, 2, 3)
- Check DroidCam connection
- Verify camera permissions

### Error Messages

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Use CPU mode or reduce resolution |
| `Camera not found` | Check camera connection and index |
| `Module not found` | Reinstall requirements: `pip install -r requirements.txt` |
| `Network timeout` | Check firewall and network connectivity |



### Development Setup

```bash
# Clone repository
git clone https://github.com/vivekteja1611/Vivid.git
cd Vivid

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Areas for Contribution

- **Velocity Estimation Accuracy** - Improve object speed calculations
- **Mobile App Development** - Native Android/iOS implementation
- **New Sensor Integration** - GPS, magnetometer, barometer
- **UI/UX Improvements** - Better accessibility features
- **Performance Optimization** - Faster processing, lower memory usage

## 📈 Current Limitations

### Known Issues
- **Velocity estimation accuracy is poor** (~50-200% error)
- **Only relative depth available** (not absolute distances)
- **Network dependency** for full sensor integration
- **Performance varies** with lighting conditions

### Planned Improvements
1. **Camera calibration** for accurate distance measurement
2. **Kalman filtering** for better velocity estimation
3. **Offline mode** for network-independent operation
4. **Mobile app** for direct sensor access

## 🎯 Roadmap

### Short-term 
- [ ] Fix velocity estimation accuracy
- [ ] Implement camera calibration
- [ ] Add offline mode
- [ ] Improve error handling

### Medium-term
- [ ] Native mobile application
- [ ] GPS integration for outdoor navigation
- [ ] Custom model training
- [ ] Advanced sensor fusion

### Long-term
- [ ] Indoor mapping capabilities
- [ ] Social features and community
- [ ] Integration with smart city infrastructure
- [ ] Research publication and dataset creation

## 📚 Documentation
- [API Reference](docs/Api.md)
- [Architecture Overview](docs/architecture.md)
- [Technical Report](VIVID_REPORT.pdf)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**VivekTeja Sapavath**
- Roll No: 24b1065
- Department of Computer Science
- Indian Institute of Technology Bombay
- Email: Vivektejaoppo1611@gmail.com
- GitHub: [@vivekteja1611](https://github.com/vivekteja1611)


## 🔗 Links

- [Demo Video](https://drive.google.com/file/d/1GHeKTfjtDI4gCBrlhmDpoVAJFXV0YOuU/view?usp=gmail)
- [Technical Report](VIVID_REPORT.pdf)
- [GitHub Repository](https://github.com/vivekteja1611/Vivid)
- [Presentation](https://vivekteja1611.github.io/Vivid-presentation/)

