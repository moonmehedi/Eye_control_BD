"""
Real-time Performance Monitor for Eye Tracking System
===================================================

Monitors system performance, accuracy, and provides diagnostics
for the eye-controlled mouse system.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import json
import threading
import cv2

class PerformanceMonitor:
    """Monitor and analyze eye tracking system performance."""
    
    def __init__(self, max_samples=1000):
        self.max_samples = max_samples
        
        # Performance metrics
        self.fps_history = deque(maxlen=max_samples)
        self.latency_history = deque(maxlen=max_samples)
        self.accuracy_history = deque(maxlen=max_samples)
        self.jitter_history = deque(maxlen=max_samples)
        
        # Timestamps
        self.frame_times = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        
        # Accuracy tracking
        self.target_positions = []
        self.gaze_positions = []
        self.click_accuracy = []
        
        # System state
        self.monitoring_active = False
        self.start_time = time.time()
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring_active = True
        self.start_time = time.time()
        print("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        print("Performance monitoring stopped")
    
    def record_frame_time(self):
        """Record frame processing time."""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Calculate FPS
        if len(self.frame_times) >= 2:
            fps = 1.0 / (self.frame_times[-1] - self.frame_times[-2])
            self.fps_history.append(fps)
    
    def record_processing_time(self, start_time, end_time):
        """Record processing latency."""
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        self.latency_history.append(latency)
        self.processing_times.append(end_time - start_time)
    
    def record_gaze_position(self, gaze_pos, target_pos=None):
        """Record gaze position for accuracy analysis."""
        self.gaze_positions.append(gaze_pos)
        
        if target_pos is not None:
            self.target_positions.append(target_pos)
            # Calculate accuracy (distance from target)
            distance = np.linalg.norm(np.array(gaze_pos) - np.array(target_pos))
            self.accuracy_history.append(distance)
    
    def calculate_jitter(self):
        """Calculate gaze jitter (stability metric)."""
        if len(self.gaze_positions) < 10:
            return 0
        
        recent_positions = np.array(list(self.gaze_positions)[-10:])
        mean_pos = np.mean(recent_positions, axis=0)
        distances = [np.linalg.norm(pos - mean_pos) for pos in recent_positions]
        jitter = np.std(distances)
        self.jitter_history.append(jitter)
        return jitter
    
    def get_current_stats(self):
        """Get current performance statistics."""
        stats = {
            "fps": {
                "current": self.fps_history[-1] if self.fps_history else 0,
                "average": np.mean(self.fps_history) if self.fps_history else 0,
                "min": np.min(self.fps_history) if self.fps_history else 0,
                "max": np.max(self.fps_history) if self.fps_history else 0
            },
            "latency_ms": {
                "current": self.latency_history[-1] if self.latency_history else 0,
                "average": np.mean(self.latency_history) if self.latency_history else 0,
                "min": np.min(self.latency_history) if self.latency_history else 0,
                "max": np.max(self.latency_history) if self.latency_history else 0
            },
            "accuracy_pixels": {
                "current": self.accuracy_history[-1] if self.accuracy_history else 0,
                "average": np.mean(self.accuracy_history) if self.accuracy_history else 0,
                "min": np.min(self.accuracy_history) if self.accuracy_history else 0,
                "max": np.max(self.accuracy_history) if self.accuracy_history else 0
            },
            "jitter_pixels": {
                "current": self.jitter_history[-1] if self.jitter_history else 0,
                "average": np.mean(self.jitter_history) if self.jitter_history else 0
            },
            "uptime_seconds": time.time() - self.start_time
        }
        return stats
    
    def save_performance_report(self, filename="performance_report.json"):
        """Save detailed performance report."""
        stats = self.get_current_stats()
        
        # Add detailed data
        report = {
            "summary": stats,
            "detailed_data": {
                "fps_history": list(self.fps_history),
                "latency_history": list(self.latency_history),
                "accuracy_history": list(self.accuracy_history),
                "jitter_history": list(self.jitter_history)
            },
            "analysis": self.analyze_performance(),
            "timestamp": time.time()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"Performance report saved to {filename}")
        except Exception as e:
            print(f"Error saving performance report: {e}")
    
    def analyze_performance(self):
        """Analyze performance and provide insights."""
        analysis = {}
        
        # FPS analysis
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            analysis["fps_rating"] = "Excellent" if avg_fps > 25 else "Good" if avg_fps > 15 else "Poor"
            analysis["fps_stability"] = "Stable" if np.std(self.fps_history) < 5 else "Unstable"
        
        # Latency analysis
        if self.latency_history:
            avg_latency = np.mean(self.latency_history)
            analysis["latency_rating"] = "Excellent" if avg_latency < 50 else "Good" if avg_latency < 100 else "Poor"
        
        # Accuracy analysis
        if self.accuracy_history:
            avg_accuracy = np.mean(self.accuracy_history)
            analysis["accuracy_rating"] = "Excellent" if avg_accuracy < 50 else "Good" if avg_accuracy < 100 else "Poor"
        
        # Jitter analysis
        if self.jitter_history:
            avg_jitter = np.mean(self.jitter_history)
            analysis["stability_rating"] = "Excellent" if avg_jitter < 10 else "Good" if avg_jitter < 25 else "Poor"
        
        return analysis
    
    def plot_performance_graphs(self):
        """Create performance visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Eye Tracking System Performance', fontsize=16)
        
        # FPS plot
        if self.fps_history:
            axes[0, 0].plot(list(self.fps_history))
            axes[0, 0].set_title('Frames Per Second')
            axes[0, 0].set_ylabel('FPS')
            axes[0, 0].grid(True)
        
        # Latency plot
        if self.latency_history:
            axes[0, 1].plot(list(self.latency_history))
            axes[0, 1].set_title('Processing Latency')
            axes[0, 1].set_ylabel('Latency (ms)')
            axes[0, 1].grid(True)
        
        # Accuracy plot
        if self.accuracy_history:
            axes[1, 0].plot(list(self.accuracy_history))
            axes[1, 0].set_title('Tracking Accuracy')
            axes[1, 0].set_ylabel('Error (pixels)')
            axes[1, 0].grid(True)
        
        # Jitter plot
        if self.jitter_history:
            axes[1, 1].plot(list(self.jitter_history))
            axes[1, 1].set_title('Gaze Stability (Jitter)')
            axes[1, 1].set_ylabel('Jitter (pixels)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('performance_graphs.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Performance graphs saved as 'performance_graphs.png'")

class SystemDiagnostics:
    """Diagnose system issues and provide recommendations."""
    
    def __init__(self):
        self.diagnostics_history = []
    
    def run_camera_diagnostics(self, camera):
        """Test camera functionality and settings."""
        diagnostics = {
            "camera_accessible": False,
            "resolution": None,
            "fps": None,
            "auto_exposure": None,
            "recommendations": []
        }
        
        try:
            # Test camera access
            ret, frame = camera.read()
            diagnostics["camera_accessible"] = ret
            
            if ret:
                diagnostics["resolution"] = f"{frame.shape[1]}x{frame.shape[0]}"
                
                # Test FPS
                start_time = time.time()
                frame_count = 0
                while time.time() - start_time < 1.0:
                    ret, _ = camera.read()
                    if ret:
                        frame_count += 1
                diagnostics["fps"] = frame_count
                
                # Check auto exposure
                auto_exp = camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                diagnostics["auto_exposure"] = auto_exp
                
                # Recommendations
                if frame_count < 20:
                    diagnostics["recommendations"].append("Low FPS detected. Check camera settings and system load.")
                if auto_exp > 0.5:
                    diagnostics["recommendations"].append("Auto exposure enabled. Consider disabling for stability.")
                if frame.shape[1] < 1280:
                    diagnostics["recommendations"].append("Low resolution detected. Increase for better accuracy.")
        
        except Exception as e:
            diagnostics["error"] = str(e)
            diagnostics["recommendations"].append("Camera initialization failed. Check camera drivers.")
        
        return diagnostics
    
    def run_lighting_diagnostics(self, frame):
        """Analyze lighting conditions."""
        diagnostics = {
            "brightness": None,
            "contrast": None,
            "uniformity": None,
            "recommendations": []
        }
        
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness (mean intensity)
            brightness = np.mean(gray)
            diagnostics["brightness"] = brightness
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray)
            diagnostics["contrast"] = contrast
            
            # Calculate uniformity (coefficient of variation)
            uniformity = contrast / brightness if brightness > 0 else 0
            diagnostics["uniformity"] = uniformity
            
            # Recommendations
            if brightness < 80:
                diagnostics["recommendations"].append("Low brightness detected. Improve lighting conditions.")
            elif brightness > 200:
                diagnostics["recommendations"].append("High brightness detected. Reduce lighting or adjust exposure.")
            
            if contrast < 30:
                diagnostics["recommendations"].append("Low contrast detected. Improve lighting setup.")
            
            if uniformity > 0.8:
                diagnostics["recommendations"].append("Uneven lighting detected. Use diffused lighting.")
        
        return diagnostics
    
    def run_face_detection_diagnostics(self, results):
        """Analyze face detection quality."""
        diagnostics = {
            "face_detected": False,
            "landmark_count": 0,
            "detection_confidence": None,
            "recommendations": []
        }
        
        if results and results.multi_face_landmarks:
            diagnostics["face_detected"] = True
            diagnostics["landmark_count"] = len(results.multi_face_landmarks[0].landmark)
            
            # Estimate confidence based on landmark consistency
            landmarks = results.multi_face_landmarks[0].landmark
            if len(landmarks) >= 468:
                diagnostics["detection_confidence"] = "Good"
            else:
                diagnostics["detection_confidence"] = "Poor"
                diagnostics["recommendations"].append("Incomplete face detection. Ensure face is fully visible.")
        else:
            diagnostics["recommendations"].append("No face detected. Check positioning and lighting.")
        
        return diagnostics
    
    def generate_diagnostic_report(self, camera_diag, lighting_diag, face_diag):
        """Generate comprehensive diagnostic report."""
        report = {
            "timestamp": time.time(),
            "camera_diagnostics": camera_diag,
            "lighting_diagnostics": lighting_diag,
            "face_detection_diagnostics": face_diag,
            "overall_recommendations": []
        }
        
        # Generate overall recommendations
        all_recommendations = []
        all_recommendations.extend(camera_diag.get("recommendations", []))
        all_recommendations.extend(lighting_diag.get("recommendations", []))
        all_recommendations.extend(face_diag.get("recommendations", []))
        
        # Remove duplicates and prioritize
        report["overall_recommendations"] = list(set(all_recommendations))
        
        # Overall system health
        issues = len(report["overall_recommendations"])
        if issues == 0:
            report["system_health"] = "Excellent"
        elif issues <= 2:
            report["system_health"] = "Good"
        elif issues <= 4:
            report["system_health"] = "Fair"
        else:
            report["system_health"] = "Poor"
        
        return report

class CalibrationValidator:
    """Validate and test calibration accuracy."""
    
    def __init__(self):
        self.test_points = []
        self.measured_points = []
    
    def add_test_point(self, target_pos, measured_pos):
        """Add a calibration test point."""
        self.test_points.append(target_pos)
        self.measured_points.append(measured_pos)
    
    def calculate_accuracy_metrics(self):
        """Calculate calibration accuracy metrics."""
        if len(self.test_points) < 2:
            return None
        
        test_points = np.array(self.test_points)
        measured_points = np.array(self.measured_points)
        
        # Calculate errors
        errors = np.linalg.norm(test_points - measured_points, axis=1)
        
        metrics = {
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
            "max_error": np.max(errors),
            "min_error": np.min(errors),
            "rms_error": np.sqrt(np.mean(errors**2)),
            "accuracy_percentage": 100 * (1 - np.mean(errors) / np.linalg.norm(np.max(test_points, axis=0) - np.min(test_points, axis=0)))
        }
        
        return metrics
    
    def generate_calibration_report(self):
        """Generate calibration quality report."""
        metrics = self.calculate_accuracy_metrics()
        
        if metrics is None:
            return {"status": "Insufficient data for validation"}
        
        # Classify accuracy
        mean_error = metrics["mean_error"]
        if mean_error < 50:
            accuracy_rating = "Excellent"
        elif mean_error < 100:
            accuracy_rating = "Good"
        elif mean_error < 200:
            accuracy_rating = "Fair"
        else:
            accuracy_rating = "Poor"
        
        report = {
            "accuracy_rating": accuracy_rating,
            "metrics": metrics,
            "recommendations": []
        }
        
        # Add recommendations
        if mean_error > 100:
            report["recommendations"].append("High tracking error. Consider recalibrating.")
        if metrics["std_error"] > 50:
            report["recommendations"].append("Inconsistent accuracy. Check head movement during calibration.")
        if metrics["max_error"] > 300:
            report["recommendations"].append("Very high maximum error. Check for outliers in calibration data.")
        
        return report

if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor()
    diagnostics = SystemDiagnostics()
    validator = CalibrationValidator()
    
    print("Performance monitoring and diagnostics system ready.")
    print("Use these classes in your main eye tracking application for comprehensive monitoring.")
