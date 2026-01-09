from flask import Flask, Response
import cv2
import mediapipe as mp
from flask_cors import CORS
import math
import time
import os

app = Flask(__name__)

# ✅ CORS: allow local + deployed frontend
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5173",
            "https://your-frontend.vercel.app"
        ]
    }
})

# -------------------------
# Health check (IMPORTANT for deployment)
# -------------------------
@app.route("/")
def home():
    return "Fitness backend is running"

# -------------------------
# MediaPipe setup
# -------------------------
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# -------------------------
# Helper utilities
# -------------------------
def calculate_angle(a, b, c):
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]
    radians = math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def landmark_dict(landmarks, img_shape):
    h, w = img_shape[:2]
    pts = {}
    for idx, lm in enumerate(landmarks.landmark):
        pts[idx] = (int(lm.x * w), int(lm.y * h))
    return pts

class PostureMonitor:
    def __init__(self, sustain_seconds=2.5):
        self.sustain_seconds = sustain_seconds
        self.bad_since = None

    def update(self, posture_ok: bool):
        now = time.time()
        if posture_ok:
            self.bad_since = None
            return False
        if self.bad_since is None:
            self.bad_since = now
            return False
        return (now - self.bad_since) >= self.sustain_seconds
    
def read_and_prepare_frame(cap, target_size=(1280,720)):
    success, img = cap.read()
    if not success or img is None:
        return success, None
    img = cv2.resize(img, target_size)
    return True, img
# -------------------------
# Endpoint: dumbbell_lateral_raise
# -------------------------
@app.route('/dumbbell_lateral_raise', methods=['GET','POST'])
def dumbbell_lateral_raise():
    cap = cv2.VideoCapture(0)
    up = False
    counter = 0
    last_count_change_time = time.time()
    posture_monitor = PostureMonitor(sustain_seconds=2.5)

    # thresholds (tune for your camera)
    RAISE_DELTA = 25          # pixels: wrist must be this much higher (smaller y) than shoulder
    SHOULDER_LEVEL = 45       # allowable pixel difference between shoulders
    TORSO_ANGLE_TOL = 22      # degrees

    if not cap.isOpened():
        return "Error: Cannot open webcam."

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.resize(img, (1280, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        show_bad_message = False

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            points = landmark_dict(results.pose_landmarks, img.shape)

            needed = [mpPose.PoseLandmark.LEFT_SHOULDER.value,
                      mpPose.PoseLandmark.RIGHT_SHOULDER.value,
                      mpPose.PoseLandmark.LEFT_WRIST.value,
                      mpPose.PoseLandmark.RIGHT_WRIST.value,
                      mpPose.PoseLandmark.LEFT_HIP.value,
                      mpPose.PoseLandmark.RIGHT_HIP.value]
            if all(n in points for n in needed):
                L_SH = mpPose.PoseLandmark.LEFT_SHOULDER.value
                R_SH = mpPose.PoseLandmark.RIGHT_SHOULDER.value
                L_WR = mpPose.PoseLandmark.LEFT_WRIST.value
                R_WR = mpPose.PoseLandmark.RIGHT_WRIST.value
                L_HIP = mpPose.PoseLandmark.LEFT_HIP.value
                R_HIP = mpPose.PoseLandmark.RIGHT_HIP.value

                # circles for debugging as before (optional)
                cv2.circle(img, points[R_SH], 8, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[R_WR], 8, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[L_SH], 8, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, points[L_WR], 8, (255, 0, 0), cv2.FILLED)

                # rep detection
                left_up = points[L_WR][1] + RAISE_DELTA < points[L_SH][1]
                right_up = points[R_WR][1] + RAISE_DELTA < points[R_SH][1]

                if not up and (left_up or right_up):
                    up = True
                    # don't increment here — you were incrementing in original code when detected UP and incremented; keep behavior
                    counter += 1
                    last_count_change_time = time.time()
                elif up and (not left_up and not right_up):
                    up = False

                # posture checks: shoulders level & torso vertical
                shoulder_diff = abs(points[L_SH][1] - points[R_SH][1])
                shoulder_ok = shoulder_diff < SHOULDER_LEVEL

                mid_shoulder = ((points[L_SH][0] + points[R_SH][0]) / 2, (points[L_SH][1] + points[R_SH][1]) / 2)
                mid_hip = ((points[L_HIP][0] + points[R_HIP][0]) / 2, (points[L_HIP][1] + points[R_HIP][1]) / 2)
                vertical_ref = (mid_shoulder[0], mid_shoulder[1] - 10)
                torso_angle = calculate_angle(vertical_ref, mid_shoulder, mid_hip)
                torso_ok = abs(torso_angle) < TORSO_ANGLE_TOL

                posture_ok = shoulder_ok and torso_ok

                # Show wrong posture only if both conditions met:
                # 1) posture been bad for sustain_seconds (posture_monitor.update True)
                # 2) no rep increment in that same duration
                posture_bad_sustained = posture_monitor.update(posture_ok)
                no_rep_since = (time.time() - last_count_change_time) >= posture_monitor.sustain_seconds

                if posture_bad_sustained and no_rep_since:
                    show_bad_message = True

                # overlays
                cv2.putText(img, str(counter), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.putText(img, f"Shoulder diff: {shoulder_diff}", (50, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                cv2.putText(img, f"Torso: {int(torso_angle)}", (50, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
            else:
                # not enough landmarks
                cv2.putText(img, "Detecting pose...", (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

        if show_bad_message:
            cv2.putText(img, "Posture is wrong", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        cv2.imshow('Webcam Feed', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return "Webcam feed closed"

# -------------------------
# Endpoint: sit-up_with_arms_on_chest
# -------------------------
@app.route('/sit-up_with_arms_on_chest',methods=['GET','POST'])
def sit_up_with_arms_on_chest():
    cap = cv2.VideoCapture(0)
    up = False
    counter = 0
    last_count_change_time = time.time()
    posture_monitor = PostureMonitor(sustain_seconds=2.5)

    # thresholds
    TORSO_ANGLE_TOL = 30   # tolerance for torso leaning (tune)
    SHOULDER_HIP_Y_TOL = 40  # pixel tolerance for shoulder vs hip when upright

    if not cap.isOpened():
        return "Error: Cannot open webcam."

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.resize(img, (1280, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        show_bad_message = False

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            points = landmark_dict(results.pose_landmarks, img.shape)

            # landmarks used: nose/shoulders/hips
            L_SH = mpPose.PoseLandmark.LEFT_SHOULDER.value
            R_SH = mpPose.PoseLandmark.RIGHT_SHOULDER.value
            MID_SH = None
            L_HIP = mpPose.PoseLandmark.LEFT_HIP.value
            R_HIP = mpPose.PoseLandmark.RIGHT_HIP.value

            if all(k in points for k in (L_SH, R_SH, L_HIP, R_HIP)):
                mid_sh = ((points[L_SH][0] + points[R_SH][0]) // 2, (points[L_SH][1] + points[R_SH][1]) // 2)
                mid_hip = ((points[L_HIP][0] + points[R_HIP][0]) // 2, (points[L_HIP][1] + points[R_HIP][1]) // 2)

                # rep detection (your original logic: shoulder y relative to hip y)
                if not up and mid_sh[1] > mid_hip[1]:
                    up = True
                elif up and mid_sh[1] < mid_hip[1]:
                    up = False
                    counter += 1
                    last_count_change_time = time.time()

                # posture check: while doing sit-ups, ensure torso angle not too twisted/curled
                vertical_ref = (mid_sh[0], mid_sh[1] - 10)
                torso_angle = calculate_angle(vertical_ref, mid_sh, mid_hip)
                torso_ok = abs(torso_angle) < TORSO_ANGLE_TOL

                posture_bad_sustained = posture_monitor.update(torso_ok)
                no_rep_since = (time.time() - last_count_change_time) >= posture_monitor.sustain_seconds

                if posture_bad_sustained and no_rep_since:
                    show_bad_message = True

                cv2.putText(img, str(counter), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.putText(img, f"Torso: {int(torso_angle)}", (50, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
            else:
                cv2.putText(img, "Detecting...", (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

        if show_bad_message:
            cv2.putText(img, "Posture is wrong", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        cv2.putText(img, str(counter), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('Webcam Feed', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Webcam feed closed"

# -------------------------
# Endpoint: air_bike (video)
# -------------------------
@app.route('/air_bike',methods=['GET','POST'])
def air_bike_counter():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Initialize MediaPipe Pose with higher confidence thresholds
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)

    # Initialize webcam capture
    cap = cv2.VideoCapture("Air-bike.mp4")
    rep_counter = 0
    prev_rep_detected = False

    if not cap.isOpened():
        return "Error: Cannot open webcam."

    # Function to calculate angle between three points
    def calculate_angle(a, b, c):
        radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
        angle = abs(radians * 180.0 / math.pi)
        return angle

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.resize(img, (1280, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extracting landmark points
            points = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                points[id] = (cx, cy)

            # Calculate angle between thigh and torso for each leg
            right_knee_angle = calculate_angle(points[24], points[26], points[28])
            left_knee_angle = calculate_angle(points[23], points[25], points[27])

            # Check for air bike movement
            if right_knee_angle > 160 and left_knee_angle > 160:
                if not prev_rep_detected:
                    rep_counter += 1
                    prev_rep_detected = True
                    print("Rep Detected")

            else:
                prev_rep_detected = False

            # Display rep counter on the frame
            cv2.putText(img, f"Reps: {rep_counter}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Display the frame in a window
        cv2.imshow('Webcam Feed', img)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return "Webcam feed closed"
# -------------------------
# Endpoint: alternate_heel_touchers (video)
# -------------------------
@app.route('/alternate_heel_touchers', methods=['GET','POST'])
def alternate_heeltouch_counter():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose_local = mp_pose.Pose(static_image_mode=False,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)

    cap = cv2.VideoCapture("Alternate-heel-touch.mp4")
    rep_counter = 0
    prev_rep_detected = False
    last_count_change_time = time.time()
    posture_monitor = PostureMonitor(sustain_seconds=2.5)

    if not cap.isOpened():
        return "Error: Cannot open video."

    while True:
        success, img = cap.read()
        if not success:
            break
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose_local.process(imgRGB)
        show_bad_message = False

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            # Using y-difference heuristic from your original code
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            right_distance = abs(right_wrist.y - right_shoulder.y)
            left_distance = abs(left_wrist.y - left_shoulder.y)

            if right_distance < 0.1 and left_distance < 0.1:
                if not prev_rep_detected:
                    rep_counter += 1
                    prev_rep_detected = True
                    last_count_change_time = time.time()
            else:
                prev_rep_detected = False

            # posture check: ensure shoulders level while doing heel touches
            pts = landmark_dict(results.pose_landmarks, img.shape)
            if mp_pose.PoseLandmark.LEFT_SHOULDER.value in pts and mp_pose.PoseLandmark.RIGHT_SHOULDER.value in pts:
                shoulder_diff = abs(pts[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] - pts[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1])
                shoulder_ok = shoulder_diff < 45
                posture_bad_sustained = posture_monitor.update(shoulder_ok)
                no_rep_since = (time.time() - last_count_change_time) >= posture_monitor.sustain_seconds
                if posture_bad_sustained and no_rep_since:
                    show_bad_message = True

            cv2.putText(img, f"Reps: {rep_counter}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        if show_bad_message:
            cv2.putText(img, "Posture is wrong", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        cv2.imshow('Video Feed', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Video feed closed"

# -------------------------
# Endpoint: shoulder_tap (video)
# -------------------------
@app.route('/shoulder_tap',methods=['GET','POST'])
def shoulder_tap_counter():
    mp_pose_local = mp.solutions.pose
    pose_local = mp_pose_local.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mp_drawing_local = mp.solutions.drawing_utils

    cap = cv2.VideoCapture("Shoulder-Tap.mp4")
    tap_count = 0
    tap_detected = False
    last_tapped_shoulder = None
    last_count_change_time = time.time()
    posture_monitor = PostureMonitor(sustain_seconds=2.5)

    if not cap.isOpened():
        return "Error: Cannot open webcam."

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.resize(img, (1280, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose_local.process(imgRGB)
        show_bad_message = False

        if results.pose_landmarks:
            mp_drawing_local.draw_landmarks(img, results.pose_landmarks, mp_pose_local.POSE_CONNECTIONS)

            # Extract key landmark points (using normalized coords)
            left_shoulder = results.pose_landmarks.landmark[mp_pose_local.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose_local.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = results.pose_landmarks.landmark[mp_pose_local.PoseLandmark.LEFT_ELBOW]
            right_elbow = results.pose_landmarks.landmark[mp_pose_local.PoseLandmark.RIGHT_ELBOW]

            # compute angles for tap detection (approx)
            def _calc(a, b, c):
                return calculate_angle([a.x, a.y], [b.x, b.y], [c.x, c.y])

            left_angle = _calc(left_shoulder, left_elbow, right_elbow)
            right_angle = _calc(right_shoulder, right_elbow, left_elbow)

            if left_angle < 100 and right_angle < 100:
                if not tap_detected:
                    if last_tapped_shoulder == 'right':
                        last_tapped_shoulder = 'left'
                    elif last_tapped_shoulder == 'left':
                        last_tapped_shoulder = 'right'
                    else:
                        last_tapped_shoulder = 'right'
                    tap_detected = True
                    tap_count += 1
                    last_count_change_time = time.time()
            else:
                tap_detected = False

            # posture check: shoulders roughly level
            pts = landmark_dict(results.pose_landmarks, img.shape)
            if (mp_pose_local.PoseLandmark.LEFT_SHOULDER.value in pts and
                mp_pose_local.PoseLandmark.RIGHT_SHOULDER.value in pts):
                shoulder_diff = abs(pts[mp_pose_local.PoseLandmark.LEFT_SHOULDER.value][1] - pts[mp_pose_local.PoseLandmark.RIGHT_SHOULDER.value][1])
                shoulder_ok = shoulder_diff < 45
                posture_bad_sustained = posture_monitor.update(shoulder_ok)
                no_rep_since = (time.time() - last_count_change_time) >= posture_monitor.sustain_seconds
                if posture_bad_sustained and no_rep_since:
                    show_bad_message = True

            cv2.putText(img, f"Shoulder Taps: {tap_count}", (50, 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        if show_bad_message:
            cv2.putText(img, "Posture is wrong", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        cv2.imshow('Shoulder Tap Counter', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Webcam feed closed"

# -------------------------
# Endpoint: dumbbell_alternate_side_press
# -------------------------
@app.route('/dumbbell_alternate_side_press',methods=['GET','POST'])
def dumbbell_alternate_side_press_counter():
    mp_pose_local = mp.solutions.pose
    pose_local = mp_pose_local.Pose(static_image_mode=False,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)
    mp_drawing_local = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)  # Change index if needed
    press_counter = 0
    prev_raised_arm = None
    last_count_change_time = time.time()
    posture_monitor = PostureMonitor(sustain_seconds=2.5)

    if not cap.isOpened():
        return "Error: Cannot open webcam."

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.resize(img, (1280, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose_local.process(imgRGB)
        show_bad_message = False

        if results.pose_landmarks:
            mp_drawing_local.draw_landmarks(img, results.pose_landmarks, mp_pose_local.POSE_CONNECTIONS)
            points = landmark_dict(results.pose_landmarks, img.shape)

            # check visibilities like your code did
            vis_ok = True
            try:
                left_elbow_confidence = results.pose_landmarks.landmark[15].visibility
                right_elbow_confidence = results.pose_landmarks.landmark[16].visibility
                left_shoulder_confidence = results.pose_landmarks.landmark[11].visibility
                right_shoulder_confidence = results.pose_landmarks.landmark[12].visibility
            except:
                left_elbow_confidence = right_elbow_confidence = left_shoulder_confidence = right_shoulder_confidence = 0
                vis_ok = False

            if vis_ok and (left_elbow_confidence > 0.7 and right_elbow_confidence > 0.7 and
                left_shoulder_confidence > 0.7 and right_shoulder_confidence > 0.7):

                left_elbow_y = points[15][1]
                right_elbow_y = points[16][1]
                left_shoulder_y = points[11][1]
                right_shoulder_y = points[12][1]

                if (left_elbow_y < left_shoulder_y and right_elbow_y > right_shoulder_y) or \
                   (left_elbow_y > left_shoulder_y and right_elbow_y < right_shoulder_y):

                    raised_arm = "left" if left_elbow_y < right_elbow_y else "right"

                    # elbow straightness check
                    left_elbow_angle = calculate_angle(points[11], points[13], points[15]) if 11 in points and 13 in points and 15 in points else 0
                    right_elbow_angle = calculate_angle(points[12], points[14], points[16]) if 12 in points and 14 in points and 16 in points else 0

                    if (raised_arm == "left" and left_elbow_angle > 160) or (raised_arm == "right" and right_elbow_angle > 160):
                        if prev_raised_arm != raised_arm:
                            press_counter += 1
                            prev_raised_arm = raised_arm
                            last_count_change_time = time.time()
            else:
                prev_raised_arm = None

            # posture check: shoulders roughly level and torso stable
            if 11 in points and 12 in points and 23 in points and 24 in points:
                shoulder_diff = abs(points[11][1] - points[12][1])
                mid_sh = ((points[11][0] + points[12][0]) / 2, (points[11][1] + points[12][1]) / 2)
                mid_hip = ((points[23][0] + points[24][0]) / 2, (points[23][1] + points[24][1]) / 2)
                vertical_ref = (mid_sh[0], mid_sh[1] - 10)
                torso_angle = calculate_angle(vertical_ref, mid_sh, mid_hip)
                shoulder_ok = shoulder_diff < 45
                torso_ok = abs(torso_angle) < 25
                posture_bad_sustained = posture_monitor.update(shoulder_ok and torso_ok)
                no_rep_since = (time.time() - last_count_change_time) >= posture_monitor.sustain_seconds
                if posture_bad_sustained and no_rep_since:
                    show_bad_message = True

            cv2.putText(img, f"Dumbbell Alt Press: {press_counter}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        if show_bad_message:
            cv2.putText(img, "Posture is wrong", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        cv2.imshow('Webcam Feed', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Webcam feed closed"

# -------------------------
# Endpoint: push-up (video)
# -------------------------
@app.route('/push-up',methods=['GET','POST'])
def pushup_counter():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    def calculate_angle(a, b, c):
        radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
        angle = abs(radians * 180.0 / math.pi)
        return angle

    # Initialize MediaPipe Pose with higher confidence thresholds
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize video capture
    cap = cv2.VideoCapture("Push-up.mp4")
    if not cap.isOpened():
        return "Error: Cannot open video file."

    pushup_count = 0
    pushup_detected = False

    while True:
        success, img = cap.read()
        if not success:
            print("Error reading frame.")
            break

        img = cv2.resize(img, (1280, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract key landmark points
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate angles
            left_angle = calculate_angle([left_elbow.x, left_elbow.y], [left_shoulder.x, left_shoulder.y],
                                         [left_wrist.x, left_wrist.y])
            right_angle = calculate_angle([right_elbow.x, right_elbow.y], [right_shoulder.x, right_shoulder.y],
                                          [right_wrist.x, right_wrist.y])

            # Check for push-up movement
            if left_angle > 160 and right_angle > 160:
                if not pushup_detected:
                    pushup_count += 1
                    pushup_detected = True
                    print("Push-up Counted")
            else:
                pushup_detected = False

            # Display push-up counter on the frame
            cv2.putText(img, f"Push-ups: {pushup_count}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Display the frame in a window
        cv2.imshow('Push-up Counter', img)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return "Video playback completed"

# -------------------------
# Endpoint: jump_squat
# -------------------------
@app.route('/jump_squat',methods=['GET','POST'])
def jump_squat_counter():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose_local = mp.solutions.pose

    pose_local = mp_pose_local.Pose(static_image_mode=False,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    squat_counter = 0
    squat_down = False
    last_count_change_time = time.time()
    posture_monitor = PostureMonitor(sustain_seconds=2.5)

    if not cap.isOpened():
        return "Error: Cannot open webcam."

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.resize(img, (1280, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose_local.process(imgRGB)
        show_bad_message = False

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose_local.POSE_CONNECTIONS)
            points = landmark_dict(results.pose_landmarks, img.shape)

            needed = (26,25,23,24)  # knee/hip indices used previously
            if all(n in points for n in needed):
                left_knee_y = points[26][1]
                right_knee_y = points[25][1]
                left_hip_y = points[23][1]
                right_hip_y = points[24][1]

                if (left_knee_y > left_hip_y and right_knee_y > right_hip_y) and not squat_down:
                    squat_down = True
                elif (left_knee_y < left_hip_y and right_knee_y < right_hip_y) and squat_down:
                    squat_down = False
                    squat_counter += 1
                    last_count_change_time = time.time()

                # posture check: knees/hips alignment and torso verticality
                mid_sh = ((points[11][0] + points[12][0]) / 2, (points[11][1] + points[12][1]) / 2) if 11 in points and 12 in points else None
                mid_hip = ((points[23][0] + points[24][0]) / 2, (points[23][1] + points[24][1]) / 2) if 23 in points and 24 in points else None
                torso_ok = True
                if mid_sh and mid_hip:
                    vertical_ref = (mid_sh[0], mid_sh[1] - 10)
                    torso_angle = calculate_angle(vertical_ref, mid_sh, mid_hip)
                    torso_ok = abs(torso_angle) < 25
                posture_bad_sustained = posture_monitor.update(torso_ok)
                no_rep_since = (time.time() - last_count_change_time) >= posture_monitor.sustain_seconds
                if posture_bad_sustained and no_rep_since:
                    show_bad_message = True

                cv2.putText(img, f"Jump Squats: {squat_counter}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            else:
                cv2.putText(img, "Detecting...", (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

        if show_bad_message:
            cv2.putText(img, "Posture is wrong", (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        cv2.imshow('Webcam Feed', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Webcam feed closed"

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
