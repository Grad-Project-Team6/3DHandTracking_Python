import math

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import socket

# parameters
width, height = 1280, 720
centerx, centery, centerz = 610, 325, -45
# (610.619, 325.238, -44.762) 중심점

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

#communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 2022)

while True:
    # Get frame from the webcam
    success, img = cap.read()
    # Hands
    hands, img = detector.findHands(img)

    data = []
    #Landmark values - (x, y, z) * 21
    if hands:

        # Get the first hand detected
        hand = hands[0]
        #Get the landmark list
        lmList = hand['lmList']
        print(lmList)

        # Calculate distance and distanceCM
        x1, y1, z1 = lmList[0]
        x2, y2, z2 = lmList[5]
        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))

        A, B, C = np.polyfit([300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57],
                             [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], 2)
        dist = int(A * distance ** 2 + B * distance + C)
        print(dist)

        #allocate
        for lm in lmList:
            newX = int((lm[0] + (dist / 10 - 1) * centerx) / (dist / 10))
            newY = int(height - (lm[1] + (dist / 10 - 1) * centery) / (dist / 10))
            newZ = int((lm[2] + (dist / 10 - 1) * centerz) / (dist / 10) + dist)

            data.extend([newX, newY, newZ])

        print(data)
        # (610.619, 325.238, -44.762) 중심점
        #send thru port
        sock.sendto(str.encode(str(data)), serverAddressPort)

    # Hands
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
