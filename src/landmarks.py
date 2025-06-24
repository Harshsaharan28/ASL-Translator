import cv2
import mediapipe as mp
import numpy as np
import csv
import os

csv_filename = 'data/landmark_data.csv'

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(header)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Starting landmark collection...")
print("Press the corresponding key (A-Z) to label the gesture.")
print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Data Collection - Press A-Z', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  
        break

    if 65 <= key <= 90 or 97 <= key <= 122:  
        label = chr(key).upper()
        if label in labels and results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                with open(csv_filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label] + landmarks)
                print(f"Saved: {label}")

cap.release()
cv2.destroyAllWindows()
