import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ======================
# EXERCISE CONFIGURATION
# ======================
EXERCISES = {
    "bicep_curl": {
        "landmarks": {
            "joint1": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "joint2": mp_pose.PoseLandmark.LEFT_ELBOW,
            "joint3": mp_pose.PoseLandmark.LEFT_WRIST
        },
        "good_angles": (30, 160),
        "feedback_rules": {
            "elbow_stability": {"threshold": 0.1, "message": "Keep elbows stable!"},
            "wrist_alignment": {"threshold": 0.2, "message": "Don't bend wrists!"}
        }
    },
    "squat": {
        "landmarks": {
            "hip": mp_pose.PoseLandmark.LEFT_HIP,
            "knee": mp_pose.PoseLandmark.LEFT_KNEE,
            "ankle": mp_pose.PoseLandmark.LEFT_ANKLE
        },
        "good_angles": (80, 120),
        "feedback_rules": {
            "knee_over_toes": {"threshold": 0.15, "message": "Knees over toes!"},
            "back_angle": {"threshold": 0.2, "message": "Maintain neutral spine!"}
        }
    }
}

CURRENT_EXERCISE = "bicep_curl"  # Change this to switch exercises

# ======================
# CORE LOGIC
# ======================
class ExerciseAnalyzer:
    def __init__(self):
        self.feedback = []
        self.rep_count = 0

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(np.degrees(radians))
        return angle if angle <= 180 else 360 - angle

    def check_form(self, landmarks, exercise_config):
        self.feedback = []
        config = EXERCISES[exercise_config]
        
        # Get coordinates
        joints = {name: [landmarks[joint.value].x, landmarks[joint.value].y] 
                 for name, joint in config["landmarks"].items()}
        
        # Calculate main angle
        angle = self.calculate_angle(joints["joint1"], joints["joint2"], joints["joint3"])
        
        # Angle-based feedback
        if not (config["good_angles"][0] <= angle <= config["good_angles"][1]):
            self.feedback.append(f"Adjust angle! Current: {angle:.1f}°")
        
        # Exercise-specific checks
        if exercise_config == "bicep_curl":
            if abs(joints["joint1"][0] - joints["joint3"][0]) > 0.1:
                self.feedback.append(config["feedback_rules"]["elbow_stability"]["message"])
        
        elif exercise_config == "squat":
            # Add knee-over-toes check
            if abs(joints["knee"][0] - joints["ankle"][0]) > 0.15:
                self.feedback.append(config["feedback_rules"]["knee_over_toes"]["message"])
        
        return angle, self.feedback

# ======================
# MAIN EXECUTION
# ======================
analyzer = ExerciseAnalyzer()
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Process image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            try:
                # Get angle and feedback
                angle, feedback = analyzer.check_form(
                    results.pose_landmarks.landmark, 
                    CURRENT_EXERCISE
                )
                
                # Display feedback
                y_offset = 30
                for msg in feedback:
                    cv2.putText(image, msg, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    y_offset += 30
                
                # Display angle
                cv2.putText(image, f"Angle: {angle:.1f}°", (10, y_offset+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            except Exception as e:
                print(f"Tracking Error: {e}")

        cv2.imshow('AI Gym Trainer', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
