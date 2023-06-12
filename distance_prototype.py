import math

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import socket

# parameters
width, height = 1280, 720

# for distance calculation
A, B, C = np.polyfit([680, 480, 360, 280, 240], [10, 15, 20, 25, 30], 2)

# for distant hand
cx, cy, cz = 633, 281, -52
# cxtemp, cytemp, cztemp = 0, 0, 0

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

# communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 2022)

while True:
    # Get frame from the webcam
    success, img = cap.read()
    # Hands
    hands, img = detector.findHands(img)

    data = []
    # Landmark values - (x, y, z) * 21
    if hands:
        # Get the first hand detected
        hand = hands[0]
        # Get the landmark list
        lmList = hand['lmList']
        # print(lmList)

        # Calculate distance and distanceCM
        x1, y1, z1 = lmList[0]
        x2, y2, z2 = lmList[5]
        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        ratio = (A * distance ** 2 + B * distance + C)/30

        print("distance: ", ratio*30)
        # print("distance: ", distanceCM)



        for lm in lmList:
            xCordData = lm[0] * (ratio ** 3)
            yCordData = height - lm[1] * (ratio ** 3)
            zCordData = lm[2] + ratio * 1000 - 1000

            data.extend([xCordData, yCordData, zCordData])

        # send thru port
        sock.sendto(str.encode(str(data)), serverAddressPort)


    # Hands
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
