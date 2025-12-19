import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import csv
import os
import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
img_size = 300

class_name = 'A'  # change this for B, C, D, etc.
csv_file = f'Data/{class_name}.csv'
os.makedirs("Data", exist_ok=True)

# Create CSV file with header if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
        writer.writerow(header)

counter = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crop the hand region from the webcam image
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(x + w + offset, img.shape[1]), min(y + h + offset, img.shape[0])
        img_crop = img[y1:y2, x1:x2]

        # White background square
        img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
        h_crop, w_crop = img_crop.shape[:2]
        aspect_ratio = w_crop / h_crop

        if aspect_ratio > 1:
            new_w = img_size
            new_h = int(img_size / aspect_ratio)
        else:
            new_h = img_size
            new_w = int(img_size * aspect_ratio)

        img_resized = cv2.resize(img_crop, (new_w, new_h))
        x_offset = (img_size - new_w) // 2
        y_offset = (img_size - new_h) // 2
        img_white[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

        # Show cropped and white-square images
        cv2.imshow("image_crop", img_crop)
        cv2.imshow("image_white", img_white)

        # Get landmarks and save when 's' is pressed
        lm_list = hand['lmList']
        landmark_data = [coord for point in lm_list for coord in point[:2]]  # x and y only

        if len(landmark_data) == 42:
            key = cv2.waitKey(1)
            if key == ord("s"):
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmark_data)
                    counter += 1
                    print(f"Saved sample {counter} for class {class_name}")

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()