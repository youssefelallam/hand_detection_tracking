import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands()

hand_draw = mp.solutions.drawing_utils

while True:
    st, frame = cap.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks is not None:
        for hand in result.multi_hand_landmarks:
            hand_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)
            
    cv2.imshow('hand detection',frame)

    if cv2.waitKey(10) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
