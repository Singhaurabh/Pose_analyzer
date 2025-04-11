from flask import Flask, render_template
import cv2
import mediapipe as mp
import numpy as np
from pose_analyzer_module import ExerciseAnalyzer

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

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
    }
}
CURRENT_EXERCISE = "bicep_curl"

@app.route('/')
def index():
    return render_template("index.html")  # Optional frontend

@app.route('/start')
def start_camera():
    cap = cv2.VideoCapture(0)
    analyzer = ExerciseAnalyzer()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                angle, feedback, reps, stage = analyzer.check_form(
                    results.pose_landmarks.landmark,
                    CURRENT_EXERCISE,
                    EXERCISES
                )

                y_offset = 30
                for msg in feedback:
                    cv2.putText(image, msg, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 30

                elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                coords = tuple(np.multiply([elbow.x, elbow.y], [640, 480]).astype(int))
                cv2.putText(image, f"{angle:.1f}°", coords, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
                cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                cv2.putText(image, str(reps), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

                cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                cv2.putText(image, str(stage), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)

                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

            cv2.imshow('AI Gym Trainer', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return "Workout session ended"

if __name__ == '__main__':
    app.run(debug=True)
