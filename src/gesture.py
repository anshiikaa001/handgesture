import cv2
import mediapipe as mp

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

# Finger tip landmark indices
finger_tips = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            lm_list = []
            h, w, c = img.shape
            for id, lm in enumerate(handLms.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            fingers = []

            # Thumb (special case: check x-axis instead of y-axis)
            if lm_list[finger_tips[0]][0] < lm_list[finger_tips[0] - 1][0]:
                fingers.append(1)  # Thumb open
            else:
                fingers.append(0)

            # Other four fingers
            for tip in finger_tips[1:]:
                if lm_list[tip][1] < lm_list[tip - 2][1]:
                    fingers.append(1)  # Finger open
                else:
                    fingers.append(0)

            # Count open fingers
            total_fingers = fingers.count(1)

            cv2.putText(img, f'Fingers: {total_fingers}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()