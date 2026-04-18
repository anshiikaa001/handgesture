
import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

tip_ids = [4, 8, 12, 16, 20]
last_action = 0

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if lm_list:
                fingers = []

                # Thumb (check left-right instead of up-down)
                if lm_list[4][0] < lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other 4 fingers
                for id in range(1, 5):
                    if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Show detection on screen
                cv2.putText(img, f"Fingers: {fingers}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                # Gestures
                if fingers == [0,0,0,0,0] and time.time() - last_action > 2:
                    cv2.putText(img, "Close Window", (10,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    pyautogui.hotkey("alt","f4")
                    last_action = time.time()

                elif fingers == [1,1,1,1,1] and time.time() - last_action > 2:
                    cv2.putText(img, "Maximize Window", (10,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    pyautogui.hotkey("win","up")
                    last_action = time.time()

                elif fingers == [0,1,1,0,0] and time.time() - last_action > 2:
                    cv2.putText(img, "Minimize Window", (10,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                    pyautogui.hotkey("win","down")
                    last_action = time.time()

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()