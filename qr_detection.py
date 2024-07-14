import cv2
import cv2.aruco as aruco
import numpy as np

# Camera parameters (assumed to be known or calibrated)
camera_matrix = np.array([[982.36, 0, 634.88],
                          [0, 981.23, 356.47],
                          [0, 0, 1]])
dist_coeffs = np.array([0.1, -0.25, 0, 0, 0])


# Function to calculate distance, yaw, pitch, roll, and vertical translation (y)
def calculate_3d_info(rvec, tvec):
    distance = np.linalg.norm(tvec)
    R, _ = cv2.Rodrigues(rvec)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    vertical_translation = tvec[1]
    return distance, yaw, pitch, roll, vertical_translation


# Function to determine the direction for the camera
def get_camera_direction(current_pose, target_pose):
    current_distance, current_yaw, current_pitch, current_roll, current_y = current_pose
    target_distance, target_yaw, target_pitch, target_roll, target_y = target_pose

    direction = None
    noise_margin = 5  # Define the noise margin

    print(f"Current Pose: Distance: {current_distance}, Yaw: {current_yaw}, Pitch: {current_pitch}, Roll: {current_roll}, Y: {current_y}")
    print(f"Target Pose: Distance: {target_distance}, Yaw: {target_yaw}, Pitch: {target_pitch}, Roll: {target_roll}, Y: {target_y}")

    # Check if current position is within the noise margin of the target position
    if (target_yaw - noise_margin <= current_yaw <= target_yaw + noise_margin and
        target_pitch - noise_margin <= current_pitch <= target_pitch + noise_margin and
        target_roll - noise_margin <= current_roll <= target_roll + noise_margin):
        return "Correct position"

    # Determine linear movement direction (forward/backward)
    if (target_yaw - noise_margin <= current_yaw <= target_yaw + noise_margin and
        target_pitch - noise_margin <= current_pitch <= target_pitch + noise_margin):
        direction = "Move forward"
        return direction

    elif (target_yaw - noise_margin <= current_yaw <= target_yaw + noise_margin):
        if current_pitch > target_pitch + noise_margin:
            direction = "Move down"
            return direction

        elif current_pitch < target_pitch - noise_margin:
            direction = "Move up"
            return direction

    elif (target_pitch - noise_margin <= current_pitch <= target_pitch + noise_margin):
        if current_yaw > target_yaw + noise_margin:
            direction = "Move right"
            return direction

        elif current_yaw < target_yaw - noise_margin:
            direction = "Move left"
            return direction

    elif (not (target_yaw - noise_margin <= current_yaw <= target_yaw + noise_margin) and
          not (target_pitch - noise_margin <= current_pitch <= target_pitch + noise_margin)):
        if current_pitch > target_pitch + noise_margin:
            direction = "Move down"
            return direction

        elif current_pitch < target_pitch - noise_margin:
            direction = "Move up"
            return direction

        if current_yaw > target_yaw + noise_margin:
            direction = "Move right"
            return direction

        elif current_yaw < target_yaw - noise_margin:
            direction = "Move left"
            return direction

    return direction



# Load the Aruco dictionary and detector parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()

# Open the camera
cap = cv2.VideoCapture(1)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press Enter to capture the initial frame...")

# Capture the initial frame
while True:
    ret, initial_frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    cv2.imshow('Initial Frame', initial_frame)

    if cv2.waitKey(1) & 0xFF == ord('\r'):  # Press Enter to capture the frame
        cv2.imwrite('initial_frame.jpg', initial_frame)
        break

cv2.destroyWindow('Initial Frame')

# Process the captured initial frame
initial_image = cv2.imread('initial_frame.jpg')
gray_initial = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
corners_initial, ids_initial, _ = aruco.detectMarkers(gray_initial, aruco_dict, parameters=parameters)

if ids_initial is not None:
    rvec_initial, tvec_initial, _ = aruco.estimatePoseSingleMarkers(corners_initial[0], 0.05, camera_matrix,
                                                                    dist_coeffs)
    initial_pose = calculate_3d_info(rvec_initial[0][0], tvec_initial[0][0])
    print("Initial pose captured.")
else:
    print("Error: No markers detected in the initial image.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Main loop to process video frames and guide the user
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], 0.05, camera_matrix, dist_coeffs)
        current_pose = calculate_3d_info(rvec[0][0], tvec[0][0])
        direction = get_camera_direction(current_pose, initial_pose)

        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)
        print(direction)

    else:
        cv2.putText(frame, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)
        print("No markers detected.")

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
