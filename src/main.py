import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Disable fail-safe (important)
pyautogui.FAILSAFE = False

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)

# Screen size
screen_w, screen_h = pyautogui.size()

# Variables
prev_x, prev_y = 0, 0
click_time = 0
last_action = 0

CLICK_DELAY = 0.8
ACTION_DELAY = 2

finger_tips = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            h, w, _ = img.shape
            lm_list = []

            for lm in handLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # ---------------- MOUSE CONTROL ----------------
            x, y = lm_list[8]

            screen_x = np.interp(x, [0, w], [0, screen_w])
            screen_y = np.interp(y, [0, h], [0, screen_h])

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / 5
            curr_y = prev_y + (screen_y - prev_y) / 5

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # ---------------- CLICK (PINCH) ----------------
            x1, y1 = lm_list[4]  # thumb
            x2, y2 = lm_list[8]  # index

            distance = np.hypot(x2 - x1, y2 - y1)

            if distance < 30 and time.time() - click_time > CLICK_DELAY:
                pyautogui.click()
                click_time = time.time()
                cv2.putText(img, "CLICK", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # ---------------- FINGER DETECTION ----------------
            fingers = []

            # Thumb
            if lm_list[4][0] < lm_list[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers
            for tip in finger_tips[1:]:
                if lm_list[tip][1] < lm_list[tip - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # ---------------- OPERATIONS ----------------
            if time.time() - last_action > ACTION_DELAY:

                # Close window (fist)
                if fingers == [0,0,0,0,0]:
                    pyautogui.hotkey("alt", "f4")
                    last_action = time.time()

                # Maximize (open hand)
                elif fingers == [1,1,1,1,1]:
                    pyautogui.hotkey("win", "up")
                    last_action = time.time()

                # Minimize (2 fingers)
                elif fingers == [0,1,1,0,0]:
                    pyautogui.hotkey("win", "down")
                    last_action = time.time()

            # ---------------- DISPLAY ----------------
            cv2.putText(img, f'Fingers: {fingers}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.circle(img, (x, y), 10, (255,0,255), cv2.FILLED)

    cv2.imshow("AI Gesture Control System", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()