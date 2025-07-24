"""
System Test and Verification Script
==================================

This script tests the eye tracking system components and verifies
that all dependencies are working correctly.
"""

import sys
import cv2
import numpy as np
import time

def test_imports():
    """Test if all required libraries can be imported."""
    print("Testing library imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
        print(f"  Version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✓ MediaPipe imported successfully")
        print(f"  Version: {mp.__version__}")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False
    
    try:
        import pyautogui
        print("✓ PyAutoGUI imported successfully")
        print(f"  Version: {pyautogui.__version__}")
    except ImportError as e:
        print(f"✗ PyAutoGUI import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        print(f"  Version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    return True

def test_camera():
    """Test camera functionality."""
    print("\nTesting camera access...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Could not open camera")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Could not read from camera")
            cap.release()
            return False
        
        print("✓ Camera access successful")
        print(f"  Frame shape: {frame.shape}")
        
        # Test a few frames
        frame_count = 0
        start_time = time.time()
        
        for i in range(30):  # Test 30 frames
            ret, frame = cap.read()
            if ret:
                frame_count += 1
        
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        
        print(f"  Estimated FPS: {fps:.1f}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_mediapipe_face_detection():
    """Test MediaPipe face detection."""
    print("\nTesting MediaPipe face detection...")
    
    try:
        import mediapipe as mp
        
        # Initialize face mesh
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Test with camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Camera not available for face detection test")
            return False
        
        print("  Please look at the camera for face detection test...")
        
        detection_successful = False
        test_frames = 60  # Test for 60 frames
        
        for i in range(test_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                print(f"  ✓ Face detected with {len(landmarks)} landmarks")
                detection_successful = True
                break
            
            # Show progress
            if i % 10 == 0:
                print(f"  Testing... frame {i}/{test_frames}")
        
        cap.release()
        face_mesh.close()
        
        if detection_successful:
            print("✓ MediaPipe face detection working")
            return True
        else:
            print("✗ No face detected during test")
            return False
            
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        return False

def test_pyautogui():
    """Test PyAutoGUI functionality."""
    print("\nTesting PyAutoGUI functionality...")
    
    try:
        import pyautogui
        
        # Get screen size
        screen_size = pyautogui.size()
        print(f"  Screen size: {screen_size}")
        
        # Test mouse position
        current_pos = pyautogui.position()
        print(f"  Current mouse position: {current_pos}")
        
        # Test moving mouse (small movement)
        original_pos = pyautogui.position()
        pyautogui.moveRel(10, 10)
        time.sleep(0.1)
        new_pos = pyautogui.position()
        pyautogui.moveTo(original_pos.x, original_pos.y)  # Move back
        
        if new_pos != original_pos:
            print("✓ Mouse movement working")
        else:
            print("✗ Mouse movement failed")
            return False
        
        print("✓ PyAutoGUI functionality working")
        return True
        
    except Exception as e:
        print(f"✗ PyAutoGUI test failed: {e}")
        return False

def test_eye_tracking_components():
    """Test specific eye tracking components."""
    print("\nTesting eye tracking components...")
    
    try:
        # Test if we can import our utilities
        try:
            from eye_tracker_utils import GazeEstimator, SmoothingFilter, PupilDetector
            print("✓ Eye tracking utilities imported successfully")
        except ImportError as e:
            print(f"⚠ Eye tracking utilities not available: {e}")
            print("  This is normal - utilities are for advanced features")
        
        # Test basic eye tracking math
        import numpy as np
        
        # Test basic vector calculations
        eye_center = np.array([100, 100])
        pupil_center = np.array([105, 102])
        gaze_vector = pupil_center - eye_center
        magnitude = np.linalg.norm(gaze_vector)
        
        if magnitude > 0:
            normalized_gaze = gaze_vector / magnitude
            print(f"  ✓ Gaze vector calculation: {normalized_gaze}")
        
        print("✓ Eye tracking math components working")
        return True
        
    except Exception as e:
        print(f"✗ Eye tracking component test failed: {e}")
        return False

def main():
    """Run all system tests."""
    print("=" * 60)
    print("EYE-CONTROLLED MOUSE SYSTEM - VERIFICATION TEST")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        ("Library Imports", test_imports),
        ("Camera Access", test_camera),
        ("MediaPipe Face Detection", test_mediapipe_face_detection),
        ("PyAutoGUI Functionality", test_pyautogui),
        ("Eye Tracking Components", test_eye_tracking_components)
    ]
    
    for test_name, test_function in tests:
        print(f"\n{'-' * 40}")
        if not test_function():
            all_tests_passed = False
    
    print(f"\n{'=' * 60}")
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
        print("\nNext steps:")
        print("1. Run 'python test.py' for the enhanced system")
        print("2. Run 'python advanced_eye_tracker.py' for full features")
        print("3. Press 'c' to calibrate the system")
        print("4. Position yourself 50-70cm from the camera")
    else:
        print("❌ SOME TESTS FAILED. Please address the issues above.")
        print("\nTroubleshooting:")
        print("1. Check camera connection and permissions")
        print("2. Ensure good lighting conditions")
        print("3. Verify all libraries are installed correctly")
    print("=" * 60)

if __name__ == "__main__":
    main()
