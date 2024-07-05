import cv2
import cv2.aruco as aruco
import numpy as np
import csv

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
    if current_yaw > target_yaw:
        directions.append("turn right")
    elif current_yaw < target_yaw:
        directions.append("turn left")

    if current_pitch > target_pitch:
        directions.append("tilt down")
    elif current_pitch < target_pitch:
        directions.append("tilt up")

    if current_roll > target_roll:
        directions.append("roll clockwise")
    elif current_roll < target_roll:
        directions.append("roll counterclockwise")

    print("1: " + str(abs(current_distance - target_distance)) +
          " 2: " + str(abs(current_yaw - target_yaw)) +
          " 3: " + str(abs(current_pitch - target_pitch)) +
          " 4: " + str(abs(current_roll - target_roll)))

    # if ((abs(current_distance - target_distance)) < 0.2):
    #     directions.append("stop move forward / backward")
    #
    # if ((abs(current_yaw - target_yaw)) < 2):
    #     directions.append("stop move right / left")
    #
    # if ((abs(current_pitch - target_pitch)) < 0.5):
    #     directions.append("stop move down / up")
    #
    # if ((abs(current_roll - target_roll)) < 4):
    #     directions.append("stop move clockwise / counterclockwise")

    if ((abs(current_distance - target_distance)) < 0.2 and
            (abs(current_yaw - target_yaw)) < 2 and (abs(current_pitch - target_pitch)) < 0.5 and (
                    abs(current_roll - target_roll)) < 4):
        directions.append("stoppppppppppppppppppp")
    return directions


# Load the Aruco dictionary and detector parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()

# Capture the initial image with the desired angle and position
initial_image = cv2.imread('p2.jpg')
gray_initial = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
corners_initial, ids_initial, _ = aruco.detectMarkers(gray_initial, aruco_dict, parameters=parameters)

if ids_initial is not None:
    rvec_initial, tvec_initial, _ = aruco.estimatePoseSingleMarkers(corners_initial[0], 0.05, camera_matrix,
                                                                    dist_coeffs)
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], 0.05, camera_matrix, dist_coeffs)
        current_pose = calculate_3d_info(rvec[0][0], tvec[0][0])
        directions = get_camera_direction(current_pose, initial_pose)

        for direction in directions:
            cv2.putText(frame, direction, (10, 30 + directions.index(direction) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            print(direction)

    else:
        print("No markers detected.")

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
