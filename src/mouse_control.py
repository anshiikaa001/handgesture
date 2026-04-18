import cv2
import mediapipe as mp
import pyautogui

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Get screen size
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror image for natural feel
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Get landmark list
            h, w, c = img.shape
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Index fingertip (id = 8)
            index_x, index_y = lm_list[8]

            # Map camera coords to screen coords
            screen_x = int(screen_w * (index_x / w))
            screen_y = int(screen_h * (index_y / h))

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Show pointer on camera
            cv2.circle(img, (index_x, index_y), 15, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Finger Mouse Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()