"""
Advanced Eye-Controlled Mouse System with Enhanced Features
==========================================================

This is an enhanced version with additional features like head pose estimation,
advanced calibration, and improved accuracy for assistive technology.
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import json
import os
from collections import deque
from eye_tracker_utils import (
    GazeEstimator, SmoothingFilter, PupilDetector, 
    CalibrationManager, calculate_eye_aspect_ratio
)

class AdvancedEyeControlledMouse:
    """Enhanced eye-controlled mouse with advanced features."""
    
    def __init__(self, config_file="eye_tracker_config.json"):
        """Initialize the advanced eye tracking system."""
        self.config_file = config_file
        self.load_configuration()
        
        # Initialize components
        self.setup_camera()
        self.setup_mediapipe()
        self.setup_screen()
        self.setup_tracking_components()
        
        # Performance metrics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print("Advanced Eye-Controlled Mouse System Initialized")
        self.print_controls()
    
    def load_configuration(self):
        """Load configuration from JSON file."""
        default_config = {
            "camera": {
                "width": 1280,
                "height": 720,
                "fps": 30,
                "device_id": 0
            },
            "tracking": {
                "smoothing_alpha": 0.3,
                "blink_threshold": 0.25,
                "click_cooldown": 1.0,
                "sensitivity": 1.0
            },
            "calibration": {
                "grid_size": 3,
                "samples_per_point": 30,
                "point_display_time": 3.0
            },
            "display": {
                "show_landmarks": True,
                "show_gaze_vectors": True,
                "show_fps": True
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    for key in default_config:
                        if key in loaded_config:
                            default_config[key].update(loaded_config[key])
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
        
        self.config = default_config
        self.save_configuration()
    
    def save_configuration(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def setup_camera(self):
        """Initialize camera with optimal settings."""
        self.cam = cv2.VideoCapture(self.config["camera"]["device_id"])
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera"]["width"])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"]["height"])
        self.cam.set(cv2.CAP_PROP_FPS, self.config["camera"]["fps"])
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure for stability
        
        # Test camera
        ret, _ = self.cam.read()
        if not ret:
            raise RuntimeError("Could not initialize camera")
    
    def setup_mediapipe(self):
        """Initialize MediaPipe with optimized settings."""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Define eye landmark indices
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Iris landmarks (more precise)
        self.LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
    
    def setup_screen(self):
        """Initialize screen and coordinate systems."""
        self.screen_w, self.screen_h = pyautogui.size()
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.001
    
    def setup_tracking_components(self):
        """Initialize tracking and processing components."""
        self.gaze_estimator = GazeEstimator()
        self.pupil_detector = PupilDetector()
        self.calibration_manager = CalibrationManager(self.screen_w, self.screen_h)
        
        # Smoothing filters
        self.gaze_smoother = SmoothingFilter()
        self.gaze_history = deque(maxlen=10)
        
        # State variables
        self.calibration_mode = False
        self.calibration_point_index = 0
        self.calibration_samples = []
        self.calibration_start_time = 0
        self.transformation_params = None
        
        # Click detection
        self.last_click_time = 0
        self.blink_counter = 0
        self.ear_history = deque(maxlen=5)
        
        # Previous values for smoothing
        self.prev_gaze_point = None
        self.prev_screen_coords = None
    
    def extract_eye_landmarks_enhanced(self, landmarks, frame_w, frame_h, eye_indices):
        """Extract eye landmarks with enhanced precision."""
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks):
                x = landmarks[idx].x * frame_w
                y = landmarks[idx].y * frame_h
                eye_points.append([x, y])
        return np.array(eye_points, dtype=np.float32)
    
    def extract_iris_center(self, landmarks, frame_w, frame_h, iris_indices):
        """Extract iris center from MediaPipe iris landmarks."""
        if not iris_indices or len(landmarks) <= max(iris_indices):
            return None
        
        iris_points = []
        for idx in iris_indices:
            x = landmarks[idx].x * frame_w
            y = landmarks[idx].y * frame_h
            iris_points.append([x, y])
        
        iris_points = np.array(iris_points)
        return np.mean(iris_points, axis=0)
    
    def calculate_enhanced_gaze(self, left_eye, right_eye, left_iris, right_iris):
        """Calculate gaze using both eye contours and iris positions."""
        if left_iris is None or right_iris is None:
            # Fallback to geometric center
            left_center = np.mean(left_eye, axis=0)
            right_center = np.mean(right_eye, axis=0)
        else:
            left_center = left_iris
            right_center = right_iris
        
        # Calculate eye centers (bounding box centers)
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        
        # Calculate relative gaze directions
        left_gaze = (left_center - left_eye_center) * self.config["tracking"]["sensitivity"]
        right_gaze = (right_center - right_eye_center) * self.config["tracking"]["sensitivity"]
        
        # Combine gaze vectors
        combined_gaze = (left_gaze + right_gaze) / 2
        combined_position = (left_eye_center + right_eye_center) / 2
        
        # Calculate final gaze point
        gaze_point = combined_position + combined_gaze * 100  # Scale factor
        
        return gaze_point, left_gaze, right_gaze
    
    def apply_advanced_smoothing(self, gaze_point):
        """Apply multiple smoothing techniques."""
        # Exponential smoothing
        if self.prev_gaze_point is not None:
            gaze_point = self.gaze_smoother.exponential_smoothing(
                gaze_point, self.prev_gaze_point, self.config["tracking"]["smoothing_alpha"]
            )
        
        self.prev_gaze_point = gaze_point.copy()
        
        # Add to history for moving average
        self.gaze_history.append(gaze_point)
        
        # Apply moving average
        smoothed_point = np.mean(list(self.gaze_history), axis=0)
        
        return smoothed_point
    
    def enhanced_blink_detection(self, left_eye, right_eye):
        """Enhanced blink detection with temporal analysis."""
        left_ear = calculate_eye_aspect_ratio(left_eye[:6])  # Use first 6 points
        right_ear = calculate_eye_aspect_ratio(right_eye[:6])
        avg_ear = (left_ear + right_ear) / 2
        
        self.ear_history.append(avg_ear)
        
        # Check for blink pattern
        if len(self.ear_history) >= 3:
            recent_ears = list(self.ear_history)[-3:]
            
            # Detect sharp drop and recovery (blink pattern)
            if (recent_ears[1] < self.config["tracking"]["blink_threshold"] and
                recent_ears[0] > self.config["tracking"]["blink_threshold"] and
                recent_ears[2] > self.config["tracking"]["blink_threshold"]):
                return True
        
        return False
    
    def handle_advanced_calibration(self, gaze_point):
        """Advanced calibration with multiple samples per point."""
        current_time = time.time()
        
        if not self.calibration_mode:
            return
        
        # Check if we should move to next point
        elapsed_time = current_time - self.calibration_start_time
        
        if elapsed_time > self.config["calibration"]["point_display_time"]:
            # Move to next calibration point
            if len(self.calibration_samples) > 0:
                # Calculate average gaze point for this calibration target
                avg_gaze = np.mean(self.calibration_samples, axis=0)
                target_point = self.calibration_manager.calibration_points[self.calibration_point_index]
                self.calibration_manager.add_calibration_sample(avg_gaze, target_point)
                print(f"Calibration point {self.calibration_point_index + 1} completed")
            
            self.calibration_point_index += 1
            self.calibration_samples = []
            self.calibration_start_time = current_time
            
            if self.calibration_point_index >= len(self.calibration_manager.calibration_points):
                # Calibration complete
                self.finish_calibration()
        else:
            # Collect samples for current point
            if elapsed_time > 1.0:  # Start collecting after 1 second
                self.calibration_samples.append(gaze_point.copy())
    
    def finish_calibration(self):
        """Complete calibration and calculate transformation."""
        self.transformation_params = self.calibration_manager.calculate_transformation_polynomial()
        self.calibration_mode = False
        
        if self.transformation_params is not None:
            print("Calibration completed successfully!")
            # Save calibration data
            self.save_calibration_data()
        else:
            print("Calibration failed. Please try again.")
    
    def save_calibration_data(self):
        """Save calibration data to file."""
        if self.transformation_params is not None:
            calib_data = {
                "transformation_params": {
                    "coeffs_x": self.transformation_params[0].tolist(),
                    "coeffs_y": self.transformation_params[1].tolist(),
                    "degree": self.transformation_params[2]
                },
                "screen_resolution": [self.screen_w, self.screen_h],
                "timestamp": time.time()
            }
            
            try:
                with open("calibration_data.json", "w") as f:
                    json.dump(calib_data, f, indent=4)
                print("Calibration data saved")
            except Exception as e:
                print(f"Error saving calibration: {e}")
    
    def load_calibration_data(self):
        """Load saved calibration data."""
        try:
            with open("calibration_data.json", "r") as f:
                calib_data = json.load(f)
            
            # Check if screen resolution matches
            saved_resolution = calib_data["screen_resolution"]
            if saved_resolution == [self.screen_w, self.screen_h]:
                coeffs_x = np.array(calib_data["transformation_params"]["coeffs_x"])
                coeffs_y = np.array(calib_data["transformation_params"]["coeffs_y"])
                degree = calib_data["transformation_params"]["degree"]
                self.transformation_params = (coeffs_x, coeffs_y, degree)
                print("Calibration data loaded successfully")
                return True
            else:
                print("Screen resolution mismatch, calibration needed")
        except Exception as e:
            print(f"Could not load calibration data: {e}")
        
        return False
    
    def map_gaze_to_screen(self, gaze_point, frame_w, frame_h):
        """Map gaze point to screen coordinates using calibration."""
        if self.transformation_params is not None:
            # Use calibrated transformation
            screen_coords = self.calibration_manager.transform_gaze_to_screen(
                gaze_point, self.transformation_params
            )
            if screen_coords is not None:
                screen_x, screen_y = screen_coords
            else:
                # Fallback to simple mapping
                screen_x = int((gaze_point[0] / frame_w) * self.screen_w)
                screen_y = int((gaze_point[1] / frame_h) * self.screen_h)
        else:
            # Simple linear mapping
            screen_x = int((gaze_point[0] / frame_w) * self.screen_w)
            screen_y = int((gaze_point[1] / frame_h) * self.screen_h)
        
        # Apply additional smoothing to screen coordinates
        if self.prev_screen_coords is not None:
            screen_x = int(0.7 * screen_x + 0.3 * self.prev_screen_coords[0])
            screen_y = int(0.7 * screen_y + 0.3 * self.prev_screen_coords[1])
        
        self.prev_screen_coords = (screen_x, screen_y)
        
        # Clamp to screen boundaries
        screen_x = np.clip(screen_x, 0, self.screen_w - 1)
        screen_y = np.clip(screen_y, 0, self.screen_h - 1)
        
        return screen_x, screen_y
    
    def draw_enhanced_visualization(self, frame, left_eye, right_eye, gaze_point, left_gaze, right_gaze):
        """Draw comprehensive visualization overlay."""
        frame_h, frame_w = frame.shape[:2]
        
        # Draw eye landmarks
        if self.config["display"]["show_landmarks"]:
            for point in left_eye:
                cv2.circle(frame, tuple(point.astype(int)), 1, (0, 255, 0), -1)
            for point in right_eye:
                cv2.circle(frame, tuple(point.astype(int)), 1, (0, 255, 0), -1)
        
        # Draw gaze vectors
        if self.config["display"]["show_gaze_vectors"]:
            left_center = np.mean(left_eye, axis=0)
            right_center = np.mean(right_eye, axis=0)
            
            left_end = left_center + left_gaze * 50
            right_end = right_center + right_gaze * 50
            
            cv2.line(frame, tuple(left_center.astype(int)), tuple(left_end.astype(int)), (255, 0, 0), 2)
            cv2.line(frame, tuple(right_center.astype(int)), tuple(right_end.astype(int)), (255, 0, 0), 2)
        
        # Draw gaze point
        cv2.circle(frame, tuple(gaze_point.astype(int)), 8, (0, 0, 255), -1)
        cv2.circle(frame, tuple(gaze_point.astype(int)), 12, (0, 0, 255), 2)
        
        # Draw calibration target
        if self.calibration_mode and self.calibration_point_index < len(self.calibration_manager.calibration_points):
            target = self.calibration_manager.calibration_points[self.calibration_point_index]
            target_x = int((target[0] / self.screen_w) * frame_w)
            target_y = int((target[1] / self.screen_h) * frame_h)
            
            cv2.circle(frame, (target_x, target_y), 25, (0, 0, 255), 3)
            cv2.circle(frame, (target_x, target_y), 8, (0, 0, 255), -1)
            
            # Countdown
            elapsed = time.time() - self.calibration_start_time
            remaining = max(0, self.config["calibration"]["point_display_time"] - elapsed)
            countdown = int(remaining) + 1
            
            cv2.putText(frame, f"Look here: {countdown}", (target_x - 80, target_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw status information
        status_color = (0, 255, 255) if self.calibration_mode else (0, 255, 0)
        status_text = "CALIBRATION" if self.calibration_mode else "TRACKING"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Draw FPS
        if self.config["display"]["show_fps"]:
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (frame_w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw calibration status
        calib_status = "CALIBRATED" if self.transformation_params is not None else "NOT CALIBRATED"
        calib_color = (0, 255, 0) if self.transformation_params is not None else (0, 0, 255)
        cv2.putText(frame, calib_status, (10, frame_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 2)
        
        # Instructions
        if not self.calibration_mode:
            instructions = [
                "Controls:",
                "C - Calibrate",
                "L - Load calibration", 
                "R - Reset",
                "Q - Quit"
            ]
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, 60 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def print_controls(self):
        """Print control instructions."""
        print("\n" + "="*50)
        print("ADVANCED EYE-CONTROLLED MOUSE SYSTEM")
        print("="*50)
        print("Controls:")
        print("  C - Start calibration")
        print("  L - Load saved calibration")
        print("  R - Reset calibration")
        print("  Q - Quit system")
        print("  ESC - Emergency stop")
        print("\nCalibration Instructions:")
        print("  1. Look at each red target for 3 seconds")
        print("  2. Keep your head still during calibration")
        print("  3. Ensure good lighting conditions")
        print("="*50 + "\n")
    
    def run(self):
        """Main execution loop with enhanced features."""
        # Try to load saved calibration
        self.load_calibration_data()
        
        try:
            while True:
                ret, frame = self.cam.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Extract enhanced eye landmarks
                    left_eye = self.extract_eye_landmarks_enhanced(landmarks, frame_w, frame_h, self.LEFT_EYE_INDICES)
                    right_eye = self.extract_eye_landmarks_enhanced(landmarks, frame_w, frame_h, self.RIGHT_EYE_INDICES)
                    
                    # Extract iris centers for more accurate tracking
                    left_iris = self.extract_iris_center(landmarks, frame_w, frame_h, self.LEFT_IRIS_INDICES)
                    right_iris = self.extract_iris_center(landmarks, frame_w, frame_h, self.RIGHT_IRIS_INDICES)
                    
                    # Calculate enhanced gaze
                    gaze_point, left_gaze, right_gaze = self.calculate_enhanced_gaze(left_eye, right_eye, left_iris, right_iris)
                    
                    # Apply advanced smoothing
                    smoothed_gaze = self.apply_advanced_smoothing(gaze_point)
                    
                    # Handle calibration
                    if self.calibration_mode:
                        self.handle_advanced_calibration(smoothed_gaze)
                    else:
                        # Map to screen and move mouse
                        screen_x, screen_y = self.map_gaze_to_screen(smoothed_gaze, frame_w, frame_h)
                        pyautogui.moveTo(screen_x, screen_y)
                        
                        # Enhanced blink detection for clicking
                        if self.enhanced_blink_detection(left_eye, right_eye):
                            current_time = time.time()
                            if current_time - self.last_click_time > self.config["tracking"]["click_cooldown"]:
                                pyautogui.click()
                                self.last_click_time = current_time
                                print("Blink click detected!")
                    
                    # Draw enhanced visualization
                    self.draw_enhanced_visualization(frame, left_eye, right_eye, smoothed_gaze, left_gaze, right_gaze)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow('Advanced Eye Controlled Mouse', frame)
                
                # Handle key input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('c') and not self.calibration_mode:
                    self.start_calibration()
                elif key == ord('l'):
                    self.load_calibration_data()
                elif key == ord('r'):
                    self.reset_calibration()
                elif key == ord('s'):
                    self.save_configuration()
                    print("Configuration saved")
                    
        except KeyboardInterrupt:
            print("\nSystem interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def start_calibration(self):
        """Start the calibration process."""
        self.calibration_mode = True
        self.calibration_point_index = 0
        self.calibration_samples = []
        self.calibration_start_time = time.time()
        self.calibration_manager.gaze_data = []
        self.calibration_manager.screen_data = []
        print("Calibration started. Look at each target for 3 seconds.")
    
    def reset_calibration(self):
        """Reset calibration data."""
        self.transformation_params = None
        self.calibration_mode = False
        # Delete calibration file
        if os.path.exists("calibration_data.json"):
            os.remove("calibration_data.json")
        print("Calibration reset")
    
    def cleanup(self):
        """Clean up resources."""
        self.cam.release()
        cv2.destroyAllWindows()
        self.save_configuration()
        print("System shutdown complete")

if __name__ == "__main__":
    try:
        system = AdvancedEyeControlledMouse()
        system.run()
    except Exception as e:
        print(f"Failed to start system: {e}")
        cv2.destroyAllWindows()
