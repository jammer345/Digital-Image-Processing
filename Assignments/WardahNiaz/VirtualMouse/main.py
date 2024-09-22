import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy


frameR = 100  # Frame Reduction
smoothening = 8
wCam, hCam = 640, 480

ptime = 0
plocx,plocy = 0,0
clocx,clocy = 0,0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr , hScr = autopy.screen.size()
#print(wScr,hScr)

while True:
    # 1. Find the hand landmarks
    success,img =  cap.read()
    img = detector.findHands(img)
    lmList,bbx = detector.findPosition(img)

    # 2. To get the tip of the index and middle finger.
    if len(lmList) !=0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        #  print(x1,y1,x2,y2)

    # 3. Check which fingers are up
        fingers = detector.fingersUp()
       # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        # 4. Only index finger--in moving mode
        if fingers[1] == 1 and fingers[2] == 0:
        # 5. Convert coordinates
            x3 = np.interp(x1,(frameR,wCam),(0,wScr))
            y3 = np.interp(y1,(frameR,hCam),(0,hScr))

            # 6. Smoothen the values
            clocx = plocx +( x3 - plocx)/smoothening
            clocy = plocy +( y3 - plocy)/smoothening
            # 7. Move mouse
            autopy.mouse.move(wScr-clocx,clocy)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocx,plocy = clocx,clocy
        # 8. Both index and middle fingers are up-- in clicking mode
        if fingers[1] == 1 and fingers[2] == 1:

            # 9. Find distance between fingers
            length,img,Lineinfo= detector.findDistance(8,12,img)
            print(length)

            # 10. Click if the distance is short
            if length < 40:
                cv2.circle(img, (Lineinfo[4], Lineinfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. Frame rate
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

# 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)