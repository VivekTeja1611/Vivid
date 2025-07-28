import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSlider, QComboBox, QGroupBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QTextCursor, QColor, QTextCharFormat
import cv2
import numpy as np
from vivid import EnhancedObstacleDetector
from collections import defaultdict
from time import time

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detection_signal = pyqtSignal(list)
    status_signal = pyqtSignal(str)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self._run_flag = True
        self._pause_flag = False
        self.camera_source = 0  # Default camera

    def run(self):
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            self.status_signal.emit("Error: Could not open camera")
            return

        self.status_signal.emit("Video started")
        
        while self._run_flag:
            if not self._pause_flag:
                ret, frame = cap.read()
                if ret:
                    # Process frame with detector
                    processed_frame, detections, _ = self.detector.process_frame(frame)
                    
                    # Convert to RGB for display
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Emit signals
                    self.change_pixmap_signal.emit(rgb_frame)
                    self.detection_signal.emit(detections)
                    
        cap.release()
        self.status_signal.emit("Video stopped")

    def stop(self):
        self._run_flag = False
        self.wait()

    def pause(self):
        self._pause_flag = True

    def resume(self):
        self._pause_flag = False

    def set_camera(self, source):
        self.camera_source = source

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Enhanced Obstacle Detection'
        self.left = 100
        self.top = 100
        self.width = 1200
        self.height = 800
        
        # Initialize detector
        self.detector = EnhancedObstacleDetector(tcp_port=8888, udp_port=8889)
        
        # Log tracking
        self.last_log_times = defaultdict(float)
        self.log_interval = 2.0  # Minimum seconds between logs for same object
        self.last_log_entry = ""
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # Left panel (video display)
        self.video_panel = QGroupBox("Live Video")
        self.video_layout = QVBoxLayout()
        
        # Video display label
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        
        # Add widgets to video panel
        self.video_layout.addWidget(self.video_label)
        self.video_panel.setLayout(self.video_layout)
        
        # Right panel (controls)
        self.control_panel = QGroupBox("Controls")
        self.control_layout = QVBoxLayout()
        
        # Camera selection
        self.camera_label = QLabel("Camera Source:")
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Default Camera", "Camera 1", "Camera 2", "DroidCam"])
        
        # Control buttons
        self.start_btn = QPushButton("Start", self)
        self.stop_btn = QPushButton("Stop", self)
        self.pause_btn = QPushButton("Pause", self)
        
        # Settings sliders
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(1, 10)  # 1-10 meters
        self.depth_slider.setValue(5)
        self.depth_label = QLabel(f"Max Depth: {self.depth_slider.value()}m")
        
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(5)
        self.sensitivity_label = QLabel(f"Sensitivity: {self.sensitivity_slider.value()}/10")
        
        # Detection log
        self.log_label = QLabel("Detection Log:")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        # Status bar
        self.status_label = QLabel("Status: Ready")
        
        # Add widgets to control panel
        self.control_layout.addWidget(self.camera_label)
        self.control_layout.addWidget(self.camera_combo)
        self.control_layout.addWidget(self.start_btn)
        self.control_layout.addWidget(self.stop_btn)
        self.control_layout.addWidget(self.pause_btn)
        self.control_layout.addWidget(self.depth_label)
        self.control_layout.addWidget(self.depth_slider)
        self.control_layout.addWidget(self.sensitivity_label)
        self.control_layout.addWidget(self.sensitivity_slider)
        self.control_layout.addWidget(self.log_label)
        self.control_layout.addWidget(self.log_text)
        self.control_layout.addWidget(self.status_label)
        self.control_layout.addStretch()
        self.control_panel.setLayout(self.control_layout)
        
        # Add panels to main layout
        self.main_layout.addWidget(self.video_panel, 70)  # 70% width
        self.main_layout.addWidget(self.control_panel, 30)  # 30% width
        
        # Connect signals
        self.start_btn.clicked.connect(self.start_video)
        self.stop_btn.clicked.connect(self.stop_video)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.depth_slider.valueChanged.connect(self.update_depth)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        
        # Initialize video thread
        self.video_thread = VideoThread(self.detector)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.detection_signal.connect(self.update_detections)
        self.video_thread.status_signal.connect(self.update_status)
        
        # Start network servers
        self.detector.network_server.start_servers()
        
        self.show()

        # Add calibration controls
        self.calibration_group = QGroupBox("IMU Calibration")
        self.calibration_layout = QVBoxLayout()
        
        # Calibration buttons
        self.calibrate_start_btn = QPushButton("Start Calibration", self)
        self.calibrate_stop_btn = QPushButton("Stop Calibration", self)
        self.calibrate_apply_btn = QPushButton("Apply Calibration", self)
        self.calibrate_reset_btn = QPushButton("Reset Calibration", self)
        
        # Calibration status
        self.calibration_status = QLabel("Calibration: Not calibrated")
        self.calibration_progress = QLabel("Samples: 0/100")
        
        # Add to layout
        self.calibration_layout.addWidget(self.calibrate_start_btn)
        self.calibration_layout.addWidget(self.calibrate_stop_btn)
        self.calibration_layout.addWidget(self.calibrate_apply_btn)
        self.calibration_layout.addWidget(self.calibrate_reset_btn)
        self.calibration_layout.addWidget(self.calibration_status)
        self.calibration_layout.addWidget(self.calibration_progress)
        self.calibration_group.setLayout(self.calibration_layout)
        
        # Add calibration group to control panel
        self.control_layout.insertWidget(3, self.calibration_group)
        
        # Connect signals
        self.calibrate_start_btn.clicked.connect(self.start_calibration)
        self.calibrate_stop_btn.clicked.connect(self.stop_calibration)
        self.calibrate_apply_btn.clicked.connect(self.apply_calibration)
        self.calibrate_reset_btn.clicked.connect(self.reset_calibration)
        
        # Calibration timer
        self.calibration_timer = QTimer()
        self.calibration_timer.timeout.connect(self.update_calibration_status)
        self.calibration_timer.start(500)  # Update every 500ms
    
    def highlight_last_line(self):
        """Highlight the most recent log entry"""
        # Create highlight format
        highlight_format = QTextCharFormat()
        highlight_format.setBackground(QColor(173, 216, 230))  # Light blue
        
        # Move cursor to end
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Select the last line
        cursor.movePosition(QTextCursor.StartOfLine)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        
        # Apply highlighting
        cursor.mergeCharFormat(highlight_format)
        
        # Remove highlighting from previous entry
        if self.last_log_entry:
            # This is simplified - in a real app you might want to track positions
            # or use a more sophisticated approach to remove old highlights
            pass
        
        # Scroll to bottom
        self.log_text.ensureCursorVisible()
    
    def update_detections(self, detections):
        """Update detection log with new detections, avoiding duplicates"""
        if not detections:
            return
            
        current_time = time()
        new_entries = []
        
        for det in detections[:3]:  # Show top 3 detections
            obj_id = det['track_id']
            log_entry = (f"ID{obj_id}: {det['label']} {det['dist_cat']} "
                        f"{det['h_pos']} (Depth: {det['depth']:.2f}m, V: {det['velocity']:.1f})")
            
            # Only log if enough time has passed since last log for this object
            if current_time - self.last_log_times[obj_id] >= self.log_interval:
                new_entries.append(log_entry)
                self.last_log_times[obj_id] = current_time
        
        if new_entries:
            # Join multiple entries with newlines
            combined_entry = "\n".join(new_entries)
            
            # Only add if different from last entry
            if combined_entry != self.last_log_entry:
                self.log_text.append(combined_entry)
                self.last_log_entry = combined_entry
                self.highlight_last_line()
    
    def start_calibration(self):
        """Start IMU calibration procedure"""
        self.detector.guidance_3d.imu_processor.set_calibration_mode(True)
        self.log_text.append("IMU calibration started - keep device stationary")
        self.calibration_status.setText("Calibration: Collecting data...")
        self.highlight_last_line()
    
    def stop_calibration(self):
        """Stop calibration without applying"""
        self.detector.guidance_3d.imu_processor.set_calibration_mode(False)
        self.log_text.append("IMU calibration stopped")
        self.calibration_status.setText("Calibration: Stopped")
        self.highlight_last_line()
    
    def apply_calibration(self):
        """Apply collected calibration data"""
        success = self.detector.guidance_3d.imu_processor.calibration.calibrate_static()
        if success:
            self.log_text.append("IMU calibration applied successfully!")
            self.calibration_status.setText("Calibration: Applied (Static)")
            
            # Log calibration values
            accel_bias = self.detector.guidance_3d.imu_processor.calibration.accel_bias
            gyro_bias = self.detector.guidance_3d.imu_processor.calibration.gyro_bias
            self.log_text.append(f"Accel Bias: X={accel_bias[0]:.4f}, Y={accel_bias[1]:.4f}, Z={accel_bias[2]:.4f}")
            self.log_text.append(f"Gyro Bias: X={gyro_bias[0]:.4f}, Y={gyro_bias[1]:.4f}, Z={gyro_bias[2]:.4f}")
            self.highlight_last_line()
        else:
            self.log_text.append("Calibration failed - need more samples (keep device still)")
            self.calibration_status.setText("Calibration: Need more samples")
            self.highlight_last_line()
    
    def reset_calibration(self):
        """Reset all calibration values"""
        self.detector.guidance_3d.imu_processor.calibration = CalibrationManager()
        self.detector.guidance_3d.imu_processor.set_calibration_mode(False)
        self.log_text.append("IMU calibration reset")
        self.calibration_status.setText("Calibration: Reset")
        self.highlight_last_line()
    
    def update_calibration_status(self):
        """Update calibration progress"""
        if self.detector.guidance_3d.imu_processor.calibration_mode:
            samples = self.detector.guidance_3d.imu_processor.calibration.calibration_samples
            self.calibration_progress.setText(f"Samples: {samples}/100")
            
            # Visual feedback for proper positioning
            if samples < 50:
                self.calibration_status.setText("Calibration: Place device horizontally")
            elif samples < 100:
                self.calibration_status.setText("Calibration: Keep device still")
            else:
                self.calibration_status.setText("Calibration: Ready to apply")
    
    def start_video(self):
        if not self.video_thread.isRunning():
            self.video_thread.start()
            self.log_text.append("Video processing started")
            self.highlight_last_line()
    
    def stop_video(self):
        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.log_text.append("Video processing stopped")
            self.highlight_last_line()
    
    def toggle_pause(self):
        if self.video_thread.isRunning():
            if self.video_thread._pause_flag:
                self.video_thread.resume()
                self.pause_btn.setText("Pause")
                self.log_text.append("Video resumed")
            else:
                self.video_thread.pause()
                self.pause_btn.setText("Resume")
                self.log_text.append("Video paused")
            self.highlight_last_line()
    
    def update_image(self, cv_img):
        """Update the video_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.setText(f"Status: {message}")
    
    def update_depth(self, value):
        """Update max depth setting"""
        self.detector.max_depth_meters = value
        self.depth_label.setText(f"Max Depth: {value}m")
    
    def update_sensitivity(self, value):
        """Update sensitivity setting"""
        self.detector.alert_threshold = 2.0 - (value * 0.15)  # Map 1-10 to 1.85-0.5m
        self.sensitivity_label.setText(f"Sensitivity: {value}/10")
    
    def change_camera(self, index):
        """Change camera source"""
        if index == 0:  # Default camera
            self.video_thread.set_camera(0)
        elif index == 1:  # Camera 1
            self.video_thread.set_camera(1)
        elif index == 2:  # Camera 2
            self.video_thread.set_camera(2)
        elif index == 3:  # DroidCam
            self.video_thread.set_camera("http://192.168.1.100:4747/video")
        
        if self.video_thread.isRunning():
            self.log_text.append(f"Switched to camera {self.camera_combo.currentText()}")
            self.highlight_last_line()
    
    def closeEvent(self, event):
        """Clean up on window close"""
        if self.video_thread.isRunning():
            self.video_thread.stop()
        
        self.detector.network_server.stop_servers()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
