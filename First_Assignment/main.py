# Importing Required Libraries
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Initializing the camera and setting the width and height of its window
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Using HandDectector method from cvzone for detecting one hand
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Global Variables
prev_pos = None
canvas = None
image_combined = None

# Function for getting the info about hand and its landmarks
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        
        hand = hands[0] 
        lmList = hand["lmList"]  

        fingers = detector.fingersUp(hand)

        return fingers, lmList
    
    else:
        return None

# Function for drawing lines on the image
def draw(info, prev_pos, canvas):
    fingers, lmlist = info
    current_pos = None

    if fingers == [0,1,0,0,0]:
        current_pos = lmlist[8][0:2]
        if prev_pos is not None:
            cv2.line(canvas, prev_pos, current_pos, (255,0,255), 10)
        
        prev_pos = current_pos
    else:
        prev_pos = None
         
    return prev_pos
    
# Infinite loop for keeping the webcam alive for capturing video
while True:
    success, img = cap.read()
    img = cv2.flip(img,1)

    if canvas is None:
        canvas = np.zeros_like(img)
        image_combined = img.copy()

    info = getHandInfo(img)

    if info:
        fingers, lmlist = info
        print(fingers)
        prev_pos = draw(info, prev_pos, canvas)

        # Combining image and canvas so that lines can be drawn directly on the image
        image_combined = cv2.addWeighted(img, 0.75, canvas, 0.25, 0)

    cv2.imshow("Combined", image_combined)
    cv2.imshow("original", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas = np.zeros_like(img)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()