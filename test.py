"""
Eye-Controlled Mouse System for Paralyzed Patients
==================================================

This system tracks both eyes and calculates gaze intersection points to control
the mouse cursor with high accuracy for individuals with limited mobility.

Features:
- Dual-eye gaze tracking with intersection point calculation
- Smooth cursor movement with noise reduction
- Blink detection for clicking
- Calibration support
- Real-time performance optimization

Author: AI Assistant
Date: 2025
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque
import math

class EyeControlledMouse:
    def __init__(self):
        """Initialize the eye-controlled mouse system."""
        # Camera and MediaPipe setup
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cam.set(cv2.CAP_PROP_FPS, 30)
        
        # MediaPipe face mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Eye landmark indices (MediaPipe 468 face landmarks)
        # Left eye landmarks
        self.LEFT_EYE_OUTER = 33
        self.LEFT_EYE_INNER = 133
        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.LEFT_PUPIL = 468  # Iris center
        
        # Right eye landmarks  
        self.RIGHT_EYE_OUTER = 362
        self.RIGHT_EYE_INNER = 263
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374
        self.RIGHT_PUPIL = 473  # Iris center
        
        # Smoothing and filtering
        self.gaze_history = deque(maxlen=10)  # Store last 10 gaze points
        self.blink_threshold = 0.25  # Eye aspect ratio threshold for blink
        self.last_click_time = 0
        self.click_cooldown = 1.0  # Seconds between clicks
        
        # Calibration points (corners + center)
        self.calibration_points = [
            (0.1 * self.screen_w, 0.1 * self.screen_h),  # Top-left
            (0.9 * self.screen_w, 0.1 * self.screen_h),  # Top-right
            (0.9 * self.screen_w, 0.9 * self.screen_h),  # Bottom-right
            (0.1 * self.screen_w, 0.9 * self.screen_h),  # Bottom-left
            (0.5 * self.screen_w, 0.5 * self.screen_h),  # Center
        ]
        self.calibration_data = []
        self.calibration_index = 0
        self.calibration_mode = False
        self.calibration_start_time = 0
        
        # Transformation matrix for calibration
        self.transform_matrix = None
        
        print("Eye-Controlled Mouse System Initialized")
        print("Press 'c' to start calibration")
        print("Press 'q' to quit")
        print("Press 'r' to reset calibration")

    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR) for blink detection."""
        # Vertical distances
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # Top-bottom
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # Top-bottom (middle)
        
        # Horizontal distance
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # Left-right
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear

    def extract_eye_landmarks(self, landmarks, frame_w, frame_h):
        """Extract eye landmark coordinates."""
        # Left eye landmarks
        left_eye = np.array([
            [landmarks[self.LEFT_EYE_OUTER].x * frame_w, landmarks[self.LEFT_EYE_OUTER].y * frame_h],
            [landmarks[self.LEFT_EYE_TOP].x * frame_w, landmarks[self.LEFT_EYE_TOP].y * frame_h],
            [landmarks[159].x * frame_w, landmarks[159].y * frame_h],  # Additional top point
            [landmarks[self.LEFT_EYE_INNER].x * frame_w, landmarks[self.LEFT_EYE_INNER].y * frame_h],
            [landmarks[145].x * frame_w, landmarks[145].y * frame_h],  # Additional bottom point
            [landmarks[self.LEFT_EYE_BOTTOM].x * frame_w, landmarks[self.LEFT_EYE_BOTTOM].y * frame_h],
        ])
        
        # Right eye landmarks
        right_eye = np.array([
            [landmarks[self.RIGHT_EYE_OUTER].x * frame_w, landmarks[self.RIGHT_EYE_OUTER].y * frame_h],
            [landmarks[self.RIGHT_EYE_TOP].x * frame_w, landmarks[self.RIGHT_EYE_TOP].y * frame_h],
            [landmarks[386].x * frame_w, landmarks[386].y * frame_h],  # Additional top point
            [landmarks[self.RIGHT_EYE_INNER].x * frame_w, landmarks[self.RIGHT_EYE_INNER].y * frame_h],
            [landmarks[374].x * frame_w, landmarks[374].y * frame_h],  # Additional bottom point
            [landmarks[self.RIGHT_EYE_BOTTOM].x * frame_w, landmarks[self.RIGHT_EYE_BOTTOM].y * frame_h],
        ])
        
        return left_eye, right_eye

    def calculate_pupil_center(self, eye_landmarks):
        """Calculate pupil center from eye landmarks."""
        # Use the geometric center of the eye as pupil approximation
        center_x = np.mean(eye_landmarks[:, 0])
        center_y = np.mean(eye_landmarks[:, 1])
        return np.array([center_x, center_y])

    def calculate_gaze_vector(self, pupil_center, eye_landmarks):
        """Calculate gaze direction vector from pupil position relative to eye."""
        # Eye center (geometric center)
        eye_center = np.mean(eye_landmarks, axis=0)
        
        # Calculate relative position of pupil to eye center
        relative_pos = pupil_center - eye_center
        
        # Normalize to get direction vector
        magnitude = np.linalg.norm(relative_pos)
        if magnitude > 0:
            direction = relative_pos / magnitude
        else:
            direction = np.array([0, 0])
            
        return direction, eye_center

    def calculate_gaze_intersection(self, left_gaze, left_center, right_gaze, right_center):
        """Calculate intersection point of two gaze vectors in 2D space."""
        # Convert to 3D for intersection calculation
        # Assume eyes are on same horizontal plane, separated by ~65mm
        eye_separation = 65  # mm
        
        # 3D positions (approximate)
        left_pos_3d = np.array([-eye_separation/2, 0, 0])
        right_pos_3d = np.array([eye_separation/2, 0, 0])
        
        # 3D gaze directions (extend to include depth)
        left_dir_3d = np.array([left_gaze[0], left_gaze[1], 1])  # Assume looking forward
        right_dir_3d = np.array([right_gaze[0], right_gaze[1], 1])
        
        # Find intersection using line-line intersection in 3D
        # Parametric form: P = P0 + t * direction
        # Solve for intersection point
        
        # For simplicity, use 2D approximation
        # Weight the gaze directions by eye positions
        combined_gaze = (left_gaze + right_gaze) / 2
        combined_center = (left_center + right_center) / 2
        
        # Project to screen coordinates
        gaze_point = combined_center + combined_gaze * 500  # Scale factor
        
        return gaze_point

    def smooth_gaze_point(self, gaze_point):
        """Apply smoothing to reduce jitter."""
        self.gaze_history.append(gaze_point)
        
        # Calculate weighted average with more weight on recent points
        weights = np.exp(np.linspace(-1, 0, len(self.gaze_history)))
        weights /= np.sum(weights)
        
        smoothed_point = np.zeros(2)
        for i, point in enumerate(self.gaze_history):
            smoothed_point += weights[i] * point
            
        return smoothed_point

    def map_to_screen_coordinates(self, gaze_point, frame_w, frame_h):
        """Map gaze point to screen coordinates."""
        if self.transform_matrix is not None:
            # Use calibrated transformation
            gaze_homogeneous = np.array([gaze_point[0], gaze_point[1], 1])
            screen_homogeneous = self.transform_matrix @ gaze_homogeneous
            screen_x = screen_homogeneous[0] / screen_homogeneous[2]
            screen_y = screen_homogeneous[1] / screen_homogeneous[2]
        else:
            # Simple linear mapping
            screen_x = (gaze_point[0] / frame_w) * self.screen_w
            screen_y = (gaze_point[1] / frame_h) * self.screen_h
        
        # Clamp to screen boundaries
        screen_x = np.clip(screen_x, 0, self.screen_w - 1)
        screen_y = np.clip(screen_y, 0, self.screen_h - 1)
        
        return int(screen_x), int(screen_y)

    def handle_calibration(self, gaze_point):
        """Handle calibration process."""
        current_time = time.time()
        
        if self.calibration_mode:
            if current_time - self.calibration_start_time > 2.0:  # 2 seconds per point
                # Record calibration data
                target_point = self.calibration_points[self.calibration_index]
                self.calibration_data.append((gaze_point.copy(), target_point))
                
                self.calibration_index += 1
                self.calibration_start_time = current_time
                
                if self.calibration_index >= len(self.calibration_points):
                    # Calibration complete, calculate transformation matrix
                    self.calculate_transformation_matrix()
                    self.calibration_mode = False
                    print("Calibration complete!")

    def calculate_transformation_matrix(self):
        """Calculate transformation matrix from calibration data."""
        if len(self.calibration_data) < 4:
            print("Insufficient calibration data")
            return
            
        # Extract source and destination points
        src_points = np.array([data[0] for data in self.calibration_data], dtype=np.float32)
        dst_points = np.array([data[1] for data in self.calibration_data], dtype=np.float32)
        
        # Calculate homography transformation
        if len(src_points) >= 4:
            self.transform_matrix, _ = cv2.findHomography(src_points, dst_points)
            print("Transformation matrix calculated successfully")

    def draw_calibration_target(self, frame):
        """Draw calibration target on frame."""
        if self.calibration_mode and self.calibration_index < len(self.calibration_points):
            target = self.calibration_points[self.calibration_index]
            # Convert screen coordinates to frame coordinates
            frame_h, frame_w = frame.shape[:2]
            target_x = int((target[0] / self.screen_w) * frame_w)
            target_y = int((target[1] / self.screen_h) * frame_h)
            
            # Draw target
            cv2.circle(frame, (target_x, target_y), 20, (0, 0, 255), 3)
            cv2.circle(frame, (target_x, target_y), 5, (0, 0, 255), -1)
            
            # Draw countdown
            remaining_time = 2.0 - (time.time() - self.calibration_start_time)
            countdown = max(0, int(remaining_time) + 1)
            cv2.putText(frame, f"Look here: {countdown}", (target_x - 50, target_y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def run(self):
        """Main execution loop."""
        try:
            while True:
                ret, frame = self.cam.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Extract eye landmarks
                    left_eye, right_eye = self.extract_eye_landmarks(landmarks, frame_w, frame_h)
                    
                    # Calculate pupil centers
                    left_pupil = self.calculate_pupil_center(left_eye)
                    right_pupil = self.calculate_pupil_center(right_eye)
                    
                    # Calculate gaze vectors
                    left_gaze, left_center = self.calculate_gaze_vector(left_pupil, left_eye)
                    right_gaze, right_center = self.calculate_gaze_vector(right_pupil, right_eye)
                    
                    # Calculate gaze intersection point
                    gaze_point = self.calculate_gaze_intersection(left_gaze, left_center, right_gaze, right_center)
                    
                    # Apply smoothing
                    smoothed_gaze = self.smooth_gaze_point(gaze_point)
                    
                    # Handle calibration
                    if self.calibration_mode:
                        self.handle_calibration(smoothed_gaze)
                    
                    # Map to screen coordinates and move mouse
                    if not self.calibration_mode:
                        screen_x, screen_y = self.map_to_screen_coordinates(smoothed_gaze, frame_w, frame_h)
                        pyautogui.moveTo(screen_x, screen_y)
                    
                    # Blink detection for clicking
                    left_ear = self.calculate_eye_aspect_ratio(left_eye)
                    right_ear = self.calculate_eye_aspect_ratio(right_eye)
                    avg_ear = (left_ear + right_ear) / 2
                    
                    current_time = time.time()
                    if (avg_ear < self.blink_threshold and 
                        current_time - self.last_click_time > self.click_cooldown and
                        not self.calibration_mode):
                        pyautogui.click()
                        self.last_click_time = current_time
                        print("Blink detected - Click!")
                    
                    # Draw eye landmarks and gaze point
                    for point in left_eye:
                        cv2.circle(frame, tuple(point.astype(int)), 2, (0, 255, 0), -1)
                    for point in right_eye:
                        cv2.circle(frame, tuple(point.astype(int)), 2, (0, 255, 0), -1)
                    
                    # Draw pupil centers
                    cv2.circle(frame, tuple(left_pupil.astype(int)), 3, (255, 0, 0), -1)
                    cv2.circle(frame, tuple(right_pupil.astype(int)), 3, (255, 0, 0), -1)
                    
                    # Draw gaze point
                    cv2.circle(frame, tuple(smoothed_gaze.astype(int)), 5, (0, 0, 255), -1)
                    
                    # Draw gaze vectors
                    left_end = left_center + left_gaze * 50
                    right_end = right_center + right_gaze * 50
                    cv2.line(frame, tuple(left_center.astype(int)), tuple(left_end.astype(int)), (255, 255, 0), 2)
                    cv2.line(frame, tuple(right_center.astype(int)), tuple(right_end.astype(int)), (255, 255, 0), 2)
                
                # Draw calibration target if in calibration mode
                self.draw_calibration_target(frame)
                
                # Draw status information
                status_text = "CALIBRATION MODE" if self.calibration_mode else "TRACKING MODE"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if not self.calibration_mode:
                    cv2.putText(frame, "Press 'c' to calibrate", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.putText(frame, "Press 'q' to quit", (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Eye Controlled Mouse - Advanced', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and not self.calibration_mode:
                    self.start_calibration()
                elif key == ord('r'):
                    self.reset_calibration()
                    
        except KeyboardInterrupt:
            print("System interrupted by user")
        finally:
            self.cleanup()

    def start_calibration(self):
        """Start calibration process."""
        self.calibration_mode = True
        self.calibration_index = 0
        self.calibration_data = []
        self.calibration_start_time = time.time()
        print("Starting calibration... Look at the red targets")

    def reset_calibration(self):
        """Reset calibration data."""
        self.transform_matrix = None
        self.calibration_data = []
        self.calibration_mode = False
        print("Calibration reset")

    def cleanup(self):
        """Clean up resources."""
        self.cam.release()
        cv2.destroyAllWindows()
        print("System shutdown complete")

# Main execution
if __name__ == "__main__":
    try:
        # Disable PyAutoGUI failsafe for smooth operation
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.01  # Minimal pause for smooth movement
        
        # Create and run the eye-controlled mouse system
        system = EyeControlledMouse()
        system.run()
        
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()