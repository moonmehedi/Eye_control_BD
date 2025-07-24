"""
Utility functions for eye tracking and gaze estimation
=====================================================

This module provides mathematical and computer vision utilities
for accurate gaze tracking and eye movement analysis.
"""

import numpy as np
import cv2
import math
from typing import Tuple, List, Optional

class GazeEstimator:
    """Advanced gaze estimation using eye geometry and 3D modeling."""
    
    def __init__(self):
        self.eye_model_points = self._create_3d_eye_model()
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def _create_3d_eye_model(self):
        """Create a 3D model of eye landmarks."""
        # 3D eye model points (approximate eye geometry in mm)
        eye_points = np.array([
            (0.0, 0.0, 0.0),      # Eye center
            (-20.0, -5.0, -6.0),  # Left corner
            (20.0, -5.0, -6.0),   # Right corner
            (0.0, -10.0, -6.0),   # Top
            (0.0, 5.0, -6.0),     # Bottom
            (0.0, 0.0, -12.0),    # Pupil center (depth)
        ], dtype=np.float32)
        return eye_points
    
    def estimate_head_pose(self, landmarks_2d, camera_matrix, dist_coeffs):
        """Estimate head pose using solvePnP."""
        if len(landmarks_2d) < 6:
            return None, None
            
        # Use solvePnP to find rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.eye_model_points,
            landmarks_2d,
            camera_matrix,
            dist_coeffs
        )
        
        if success:
            return rotation_vector, translation_vector
        return None, None
    
    def calculate_gaze_vector_3d(self, pupil_center, eye_landmarks, head_pose):
        """Calculate 3D gaze vector considering head pose."""
        if head_pose[0] is None:
            # Fallback to 2D calculation
            return self.calculate_gaze_vector_2d(pupil_center, eye_landmarks)
        
        rotation_vector, translation_vector = head_pose
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate gaze direction in eye coordinate system
        eye_center = np.mean(eye_landmarks[:4], axis=0)  # Use outer landmarks
        pupil_displacement = pupil_center - eye_center
        
        # Normalize and extend to 3D
        gaze_2d = pupil_displacement / np.linalg.norm(pupil_displacement) if np.linalg.norm(pupil_displacement) > 0 else np.array([0, 0])
        gaze_3d = np.array([gaze_2d[0], gaze_2d[1], -1.0])  # Assume looking into screen
        
        # Apply head rotation
        gaze_world = rotation_matrix @ gaze_3d
        
        return gaze_world
    
    def calculate_gaze_vector_2d(self, pupil_center, eye_landmarks):
        """Calculate 2D gaze vector (fallback method)."""
        eye_center = np.mean(eye_landmarks, axis=0)
        gaze_vector = pupil_center - eye_center
        
        # Normalize
        magnitude = np.linalg.norm(gaze_vector)
        if magnitude > 0:
            gaze_vector = gaze_vector / magnitude
        
        return gaze_vector

class SmoothingFilter:
    """Various smoothing algorithms for gaze tracking."""
    
    @staticmethod
    def exponential_smoothing(current_value, previous_value, alpha=0.3):
        """Exponential smoothing filter."""
        if previous_value is None:
            return current_value
        return alpha * current_value + (1 - alpha) * previous_value
    
    @staticmethod
    def kalman_filter_1d(measurements, initial_state=0, initial_variance=1000):
        """1D Kalman filter for smooth tracking."""
        # Simple Kalman filter implementation
        state = initial_state
        variance = initial_variance
        process_variance = 0.01
        measurement_variance = 1.0
        
        filtered_values = []
        
        for measurement in measurements:
            # Prediction step
            predicted_state = state
            predicted_variance = variance + process_variance
            
            # Update step
            kalman_gain = predicted_variance / (predicted_variance + measurement_variance)
            state = predicted_state + kalman_gain * (measurement - predicted_state)
            variance = (1 - kalman_gain) * predicted_variance
            
            filtered_values.append(state)
        
        return filtered_values
    
    @staticmethod
    def moving_average(values, window_size=5):
        """Simple moving average filter."""
        if len(values) < window_size:
            return np.mean(values) if values else 0
        return np.mean(values[-window_size:])

class PupilDetector:
    """Advanced pupil detection algorithms."""
    
    def __init__(self):
        self.hough_params = {
            'dp': 1,
            'min_dist': 30,
            'param1': 50,
            'param2': 30,
            'min_radius': 5,
            'max_radius': 50
        }
    
    def detect_pupil_hough(self, eye_region):
        """Detect pupil using Hough Circle Transform."""
        # Convert to grayscale if needed
        if len(eye_region.shape) == 3:
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_region
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_params['dp'],
            minDist=self.hough_params['min_dist'],
            param1=self.hough_params['param1'],
            param2=self.hough_params['param2'],
            minRadius=self.hough_params['min_radius'],
            maxRadius=self.hough_params['max_radius']
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Return the most prominent circle (first one)
            return circles[0][:2]  # x, y coordinates
        
        return None
    
    def detect_pupil_contour(self, eye_region):
        """Detect pupil using contour analysis."""
        if len(eye_region.shape) == 3:
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_region
        
        # Apply threshold to isolate dark regions (pupil)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the pupil)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate centroid
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        
        return None

class CalibrationManager:
    """Manages calibration process and coordinate transformation."""
    
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_points = self._generate_calibration_points()
        self.gaze_data = []
        self.screen_data = []
        
    def _generate_calibration_points(self, grid_size=3):
        """Generate calibration points in a grid pattern."""
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i + 1) * self.screen_width // (grid_size + 1)
                y = (j + 1) * self.screen_height // (grid_size + 1)
                points.append((x, y))
        return points
    
    def add_calibration_sample(self, gaze_point, screen_point):
        """Add a calibration sample."""
        self.gaze_data.append(gaze_point)
        self.screen_data.append(screen_point)
    
    def calculate_transformation_polynomial(self, degree=2):
        """Calculate polynomial transformation from gaze to screen coordinates."""
        if len(self.gaze_data) < 6:  # Minimum points for 2nd degree polynomial
            return None
        
        gaze_points = np.array(self.gaze_data)
        screen_points = np.array(self.screen_data)
        
        # Create polynomial features
        X = self._create_polynomial_features(gaze_points[:, 0], gaze_points[:, 1], degree)
        
        # Fit polynomial for x and y coordinates separately
        try:
            # Solve for x coordinates
            coeffs_x = np.linalg.lstsq(X, screen_points[:, 0], rcond=None)[0]
            # Solve for y coordinates  
            coeffs_y = np.linalg.lstsq(X, screen_points[:, 1], rcond=None)[0]
            
            return coeffs_x, coeffs_y, degree
        except np.linalg.LinAlgError:
            return None
    
    def _create_polynomial_features(self, x, y, degree):
        """Create polynomial feature matrix."""
        features = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                features.append((x ** i) * (y ** j))
        return np.column_stack(features)
    
    def transform_gaze_to_screen(self, gaze_point, transformation_params):
        """Transform gaze coordinates to screen coordinates using polynomial."""
        if transformation_params is None:
            return None
        
        coeffs_x, coeffs_y, degree = transformation_params
        x, y = gaze_point
        
        # Create feature vector
        features = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                features.append((x ** i) * (y ** j))
        features = np.array(features)
        
        # Apply transformation
        screen_x = np.dot(coeffs_x, features)
        screen_y = np.dot(coeffs_y, features)
        
        return (int(screen_x), int(screen_y))

def calculate_eye_aspect_ratio(eye_landmarks):
    """Calculate Eye Aspect Ratio for blink detection."""
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

def estimate_focus_distance(left_gaze, right_gaze, baseline_distance=65):
    """Estimate focus distance from vergence angle."""
    # Calculate vergence angle
    vergence_angle = np.arccos(np.dot(left_gaze, right_gaze))
    
    # Estimate distance using simple trigonometry
    if vergence_angle > 0:
        distance = (baseline_distance / 2) / np.tan(vergence_angle / 2)
    else:
        distance = float('inf')  # Parallel gaze
    
    return distance

def find_gaze_intersection_3d(left_eye_pos, left_gaze_dir, right_eye_pos, right_gaze_dir):
    """Find 3D intersection point of two gaze rays."""
    # Line 1: P1 = left_eye_pos + t * left_gaze_dir
    # Line 2: P2 = right_eye_pos + s * right_gaze_dir
    
    # Find closest points on both lines
    w = left_eye_pos - right_eye_pos
    u = left_gaze_dir
    v = right_gaze_dir
    
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    
    denominator = a * c - b * b
    
    if abs(denominator) < 1e-6:
        # Lines are parallel
        return None
    
    t = (b * e - c * d) / denominator
    s = (a * e - b * d) / denominator
    
    # Calculate closest points
    point1 = left_eye_pos + t * left_gaze_dir
    point2 = right_eye_pos + s * right_gaze_dir
    
    # Return midpoint as intersection estimate
    intersection = (point1 + point2) / 2
    return intersection
