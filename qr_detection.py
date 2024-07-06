import cv2
import cv2.aruco as aruco
import numpy as np
from collections import deque

# Camera parameters (assumed to be known or calibrated)
camera_matrix = np.array([[982.36, 0, 634.88],
                          [0, 981.23, 356.47],
                          [0, 0, 1]])
dist_coeffs = np.array([0.1, -0.25, 0, 0, 0])

# Function to calculate distance, yaw, pitch, and roll
def calculate_3d_info(rvec, tvec):
    distance = np.linalg.norm(tvec)
    R, _ = cv2.Rodrigues(rvec)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return distance, yaw, pitch, roll

# Function to smooth values using a running average
def smooth_values(values, new_value, max_len=5):
    values.append(new_value)
    if len(values) > max_len:
        values.popleft()
    return sum(values) / len(values)

# Function to determine the direction for the camera
def get_camera_direction(current_pose, target_pose):
    directions = []
    current_distance, current_yaw, current_pitch, current_roll = current_pose
    target_distance, target_yaw, target_pitch, target_roll = target_pose

    # Determine linear movement direction
    if current_distance > target_distance:
        directions.append("forward")
    elif current_distance < target_distance:
        directions.append("backward")

    # Determine angular adjustments
    if current_pitch > target_pitch:
        directions.append("tilt down")
    elif current_pitch < target_pitch:
        directions.append("tilt up")

    if current_yaw > target_yaw:
        directions.append("turn right")
    elif current_yaw < target_yaw:
        directions.append("turn left")

    if current_roll > target_roll:
        directions.append("roll clockwise")
    elif current_roll < target_roll:
        directions.append("roll counterclockwise")

    print("Distance: ", abs(current_distance - target_distance),
          "Yaw: ", abs(current_yaw - target_yaw),
          "Pitch: ", abs(current_pitch - target_pitch),
          "Roll: ", abs(current_roll - target_roll))

    if (abs(current_distance - target_distance) < 0.05 and
            abs(current_yaw - target_yaw) < 0.05 and
            abs(current_pitch - target_pitch) < 0.1 and
            abs(current_roll - target_roll) < 0.2):
        directions.append("stop")

    return directions

# Load the Aruco dictionary and detector parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()

# Capture the initial image with the desired angle and position
initial_image = cv2.imread('p2.jpg')
gray_initial = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
corners_initial, ids_initial, _ = aruco.detectMarkers(gray_initial, aruco_dict, parameters=parameters)

if ids_initial is not None:
    rvec_initial, tvec_initial, _ = aruco.estimatePoseSingleMarkers(corners_initial[0], 0.05, camera_matrix, dist_coeffs)
    initial_pose = calculate_3d_info(rvec_initial[0][0], tvec_initial[0][0])
else:
    print("Error: No markers detected in the initial image.")
    exit()

# Open the video file
cap = cv2.VideoCapture(0)

# Check if the video file is opened correctly
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Initialize deques for smoothing pose values
distance_values = deque()
yaw_values = deque()
pitch_values = deque()
roll_values = deque()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], 0.05, camera_matrix, dist_coeffs)

        # Print raw rvec and tvec for debugging
        print("rvec:", rvec[0][0])
        print("tvec:", tvec[0][0])

        raw_pose = calculate_3d_info(rvec[0][0], tvec[0][0])

        # Print raw pitch value for debugging
        print("Raw pitch:", raw_pose[2])

        # Smooth the values
        smoothed_pose = (
            smooth_values(distance_values, raw_pose[0]),
            smooth_values(yaw_values, raw_pose[1]),
            smooth_values(pitch_values, raw_pose[2]),
            smooth_values(roll_values, raw_pose[3])
        )

        current_pose = smoothed_pose
        directions = get_camera_direction(current_pose, initial_pose)

        for direction in directions:
            cv2.putText(frame, direction, (10, 30 + directions.index(direction) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            print(direction)

    else:
        print("No markers detected.")

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
