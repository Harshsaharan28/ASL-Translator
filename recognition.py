import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model('sign_language_model.keras')

scaler = joblib.load('scaler.pkl')

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            input_data = np.array(landmarks).reshape(1, -1)
            input_data = scaler.transform(input_data)

            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = labels[predicted_class]

            cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
