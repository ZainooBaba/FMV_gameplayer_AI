import cv2 as cv
import numpy as np
import pyautogui
import time

def find_speech_bubbles(min_area=5000, max_area=10000):
    # Take a screenshot using pyautogui
    screenshot = pyautogui.screenshot()

    # Convert screenshot to a format that OpenCV can work with
    img = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)

    # Define the exact color to match (white color in BGR)
    target_color = np.array([255, 255, 255], dtype=np.uint8)  # White in BGR

    # Create a mask where all pixels matching the target color are white, and others are black
    mask = cv.inRange(img, target_color, target_color)
    cv.imwrite('mask.png', mask)  # Save mask image

    # Find contours in the binary mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    center_points = []  # List to store center points of speech bubbles

    # Draw filtered contours and their centers
    contours_img = img.copy()
    for contour in contours:
        area = cv.contourArea(contour)
        if min_area < area < max_area:
            # Calculate the center of the contour
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            center_points.append((cx, cy))

            # Draw the contour and center point
            cv.drawContours(contours_img, [contour], -1, (0, 255, 0), 2)
            cv.circle(contours_img, (cx, cy), 5, (0, 0, 255), -1)

    # Save the image with contours and center points
    cv.imwrite('contours_with_centers.png', contours_img)  # Save filtered contours image

    # Print the center points
    print("Center Points:", center_points)

    # Show the results
    cv.imshow('Mask', mask)
    cv.imshow('Contours with Centers', contours_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Return the center points as a tuple array
    return tuple(center_points)

# Example usage:
center_points_tuple_array = find_speech_bubbles()
