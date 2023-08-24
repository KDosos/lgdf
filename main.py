import numpy as np
import cv2

# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# Start a while loop
while True:
    # Reading the video from the webcam in image frames
    ret, imageFrame = webcam.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the imageFrame to HSV color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and green
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)

    # Create masks for red and green regions
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Split the frame into two halves vertically
    height, width, _ = imageFrame.shape
    left_half = imageFrame[:, :width // 2]
    right_half = imageFrame[:, width // 2:]

    # Apply masks to frames
    red_region_left = cv2.bitwise_and(left_half, left_half, mask=red_mask[:, :width // 2])
    green_region_left = cv2.bitwise_and(left_half, left_half, mask=green_mask[:, :width // 2])
    red_region_right = cv2.bitwise_and(right_half, right_half, mask=red_mask[:, width // 2:])
    green_region_right = cv2.bitwise_and(right_half, right_half, mask=green_mask[:, width // 2:])

    # Find contours and draw rectangles for detected colors on the left half
    contours, _ = cv2.findContours(cv2.cvtColor(red_region_left, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(left_half, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(left_half, "Red", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    contours, _ = cv2.findContours(cv2.cvtColor(green_region_left, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(left_half, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(left_half, "Green", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Find contours and draw rectangles for detected colors on the right half
    contours, _ = cv2.findContours(cv2.cvtColor(red_region_right, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(right_half, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(right_half, "Red", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    contours, _ = cv2.findContours(cv2.cvtColor(green_region_right, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(right_half, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(right_half, "Green", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frames with designated colors
    cv2.imshow("Left Half", left_half)
    cv2.imshow("Right Half", right_half)