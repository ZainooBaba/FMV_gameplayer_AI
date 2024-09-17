import os

import pyautogui
import time
import cv2 as cv
import numpy as np
from PIL import Image
import PIL

from pytesseract import pytesseract


# Move the cursor to an absolute position (x, y)
def reset_screen_position():
    pyautogui.moveTo(1420, 626)
    pyautogui.scroll(-200)
    pyautogui.drag(0, -400, 0.1, button='right')
    pyautogui.moveTo(1420, 626)
    pyautogui.drag(0, -400, 0.1, button='right')
    pyautogui.moveTo(1420, 626)
    pyautogui.drag(0, -400, 0.1, button='right')
    pyautogui.moveTo(1420, 626)
    pyautogui.drag(0, -400, 0.1, button='right')
    # print(pyautogui.position())
    pyautogui.moveTo(1420, 626)
    pyautogui.drag(0, 60, 0.4, pyautogui.easeOutQuad ,button='right')

def get_amount_of_crates():
    region = (848, 719, 59, 26)
    screenshot = pyautogui.screenshot(region=region,)
    text = pytesseract.image_to_string(screenshot, config='--psm 6')
    return text.split('/')[0]

def find_speech_bubbles(min_area=5000, max_area=7000, avoid_points = [(450, 1640),(1750,1470)],debug=False):
    screenshot = pyautogui.screenshot()
    img = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
    target_color = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv.inRange(img, target_color, target_color)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    center_points = []
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
            flag = False
            for avoid_point in avoid_points:
                if abs(cx - avoid_point[0]) < 50 and abs(cy - avoid_point[1]) < 50:
                    flag = True
                    break
            if flag:
                continue
            center_points.append((cx/2, cy/2))

            if debug:
                cv.drawContours(contours_img, [contour], -1, (0, 255, 0), 2)
                cv.circle(contours_img, (cx, cy), 20, (0, 0, 255), -1)

                for avoid_point in avoid_points:
                    cv.circle(contours_img, avoid_point, 20, (255, 0, 255), -1)

    if debug:
        cv.imwrite('contours_with_centers.png', contours_img)
        print("Center Points:", center_points)
        cv.imshow('Mask', mask)
        cv.imshow('Contours with Centers', contours_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return tuple(center_points)

def find_harvest_crops():
    return find_speech_bubbles(avoid_points=[(450, 1640),(1750,1470),(1746,1116),(1552, 1025)])

def collect_crops():
    speach_bubbles = find_harvest_crops()
    for speach_bubble in speach_bubbles:
        # before = pyautogui.screenshot()
        capture_and_save_screenshot("before.png")
        pyautogui.moveTo(speach_bubble[0], speach_bubble[1])
        pyautogui.click()
        time.sleep(1.5)
        print(speach_bubble[0], speach_bubble[1])
        print(type(speach_bubble[0]))
        # after = pyautogui.screenshot()
        capture_and_save_screenshot("after.png")
        position = find_largest_change(
            avoid_x=speach_bubble[0] * 2, avoid_y=speach_bubble[1] * 2,
            threshold_distance=100
        )
        print(position)
        pyautogui.moveTo(position[0]/2, position[1]/2)
        pyautogui.click()
        time.sleep(2)


def capture_and_save_screenshot(filename):
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    print(f"Screenshot saved as {filename}")


def load_image(filename):
    # Open the image file and convert it to OpenCV format
    image_pil = Image.open(filename)
    return cv.cvtColor(np.array(image_pil), cv.COLOR_RGB2BGR)


def find_largest_change(avoid_x: int, avoid_y: int, threshold_distance: int, bbox=(626, 240, 2252, 1316),
                        screenshot1_path='before.png', screenshot2_path='after.png', debug=False):
    # Check if screenshots already exist
    if not os.path.exists(screenshot1_path):
        print(f"{screenshot1_path} not found. Capturing screenshot 1...")
        capture_and_save_screenshot(screenshot1_path)

    if not os.path.exists(screenshot2_path):
        print(f"{screenshot2_path} not found. Capturing screenshot 2...")
        capture_and_save_screenshot(screenshot2_path)

    # Load images
    image1 = load_image(screenshot1_path)
    image2 = load_image(screenshot2_path)

    # Convert images to grayscale
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

    # Compute absolute difference and threshold
    diff = cv.absdiff(gray1, gray2)
    cv.imwrite('differ.png', diff)
    _, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Check if contour is within the avoid area
    def is_within_avoid_area(contour):
        x, y, w, h = cv.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2
        return np.sqrt((center_x - avoid_x) ** 2 + (center_y - avoid_y) ** 2) < threshold_distance

    # Check if contour is within the bounding box
    def is_within_bbox(contour, bbox):
        x, y, w, h = cv.boundingRect(contour)
        return (x >= bbox[0] and y >= bbox[1] and
                x + w <= bbox[0] + bbox[2] and y + h <= bbox[1] + bbox[3])

    # Find largest contour outside the avoid area and within the bounding box
    max_contour, max_area = None, 0
    for contour in contours:
        if not is_within_avoid_area(contour) and is_within_bbox(contour, bbox):
            area = cv.contourArea(contour)
            if area > max_area:
                max_area, max_contour = area, contour

    # Debug: Save and show images
    if debug:
        cv.rectangle(image2,bbox,(0,255,0),5)
        cv.circle(image2, (int(avoid_x), int(avoid_y)), threshold_distance, (255, 0, 0), 5)
        # Save images for debugging
        cv.imwrite('diff_image.png', diff)
        cv.imwrite('threshold_image.png', thresh)

        # Draw contours on the original image
        contours_img = cv.drawContours(image2.copy(), contours, -1, (0, 255, 0), 2)
        cv.imwrite('contours_image.png', contours_img)

        if max_contour is not None:
            x, y, w, h = cv.boundingRect(max_contour)
            cv.rectangle(image2, (x, y), (x + w, y + h), (255, 0, 255), 5)
            cv.imwrite('highlighted_changes.png', image2)


        # Show images using OpenCV
        cv.imshow('Difference Image', diff)
        cv.imshow('Threshold Image', thresh)
        cv.imshow('Contours Image', contours_img)
        if max_contour is not None:
            cv.imshow('Highlighted Changes', image2)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Return the maximum change value and its position
    if max_contour is not None:
        x, y, w, h = cv.boundingRect(max_contour)
        return (x + w // 2, y + h // 2)  # Return the center of the largest contour
    else:
        print("No significant changes found within the bounding box.")
        return None

time.sleep(1)
reset_screen_position()
collect_crops()

# Move the cursor relative to its current position
# pyautogui.moveRel(50, -50)

