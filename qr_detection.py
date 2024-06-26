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
    # Calculate distance to the camera
    distance = np.linalg.norm(tvec)

    # Calculate yaw, pitch, and roll angles
    R, _ = cv2.Rodrigues(rvec)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])

    return distance, yaw, pitch, roll


# Function to determine the direction for the drone
def get_drone_direction(distance, yaw, pitch, roll):
    direction = "continue"

    # Threshold values for direction decisions
    distance_threshold = 1.0  # Adjusted distance threshold
    yaw_threshold = 0.5  # Adjusted yaw threshold
    pitch_threshold = 0.5  # Adjusted pitch threshold
    roll_threshold = 0.5  # Adjusted roll threshold

    # Determine direction based on 3D information
    if distance > distance_threshold:
        direction = "forward"
    elif distance < -distance_threshold:
        direction = "backward"
    elif yaw > yaw_threshold:
        direction = "right"
    elif yaw < -yaw_threshold:
        direction = "left"
    elif pitch > pitch_threshold:
        direction = "up"
    elif pitch < -pitch_threshold:
        direction = "down"
    elif roll > roll_threshold:
        direction = "clockwise"
    elif roll < -roll_threshold:
        direction = "counterclockwise"

    return direction


# Load the Aruco dictionary and detector parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters()

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# CSV file to write the output
with open('aruco_output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame ID", "QR ID", "QR 2D", "Distance", "Yaw", "Pitch", "Roll", "Direction"])

    # Process each frame
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # If markers are detected
        if ids is not None:
            for i in range(len(ids)):
                # Get the corner points
                corner_points = corners[i][0].tolist()

                # Draw the detected markers with green rectangular frame
                frame = aruco.drawDetectedMarkers(frame, corners)

                # Estimate pose of each marker
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)

                # Calculate 3D information
                distance, yaw, pitch, roll = calculate_3d_info(rvec[0][0], tvec[0][0])

                # Debug prints to check the values
                #print(f"ID: {ids[i][0]}, Distance: {distance}, Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")

                # Determine the direction for the drone
                direction = get_drone_direction(distance, yaw, pitch, roll)

                # Draw the pose of the marker
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

                # Write to CSV
                writer.writerow([frame_id, ids[i][0], corner_points, distance, np.degrees(yaw), np.degrees(pitch),
                                 np.degrees(roll), direction])

                # Display the marker ID and direction on the frame
                cv2.putText(frame, f"ID: {ids[i][0]}, Dir: {direction}", tuple(map(int, corner_points[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                # Print the direction to console
                print(direction)

        else:
            # If no markers are detected, output "continue"
            direction = "continue"
            print(direction)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
