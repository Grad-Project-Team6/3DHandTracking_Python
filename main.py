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

        coord_squared = (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2
        if coord_squared < 0:
            hand_length = 0
        else:
            hand_length = math.sqrt(coord_squared)

        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        ratio = (A * distance ** 2 + B * distance + C)/30
        # print("hand_length: ", hand_length)
        # print("distance: ", ratio*30)
        # print("distance: ", distanceCM)

        # TODO: calculate center point
        # x 좌표들의 평균 계산
        x_center = sum([coord[0] for coord in lmList]) / len(lmList)

        # y 좌표들의 평균 계산
        y_center = sum([coord[1] for coord in lmList]) / len(lmList)

        # z 좌표들의 평균 계산
        z_center = sum([coord[2] for coord in lmList]) / len(lmList)

        # 기준 거리 : 38, 줄이고자 하는 크기: distance
        if hand_length < 0:
            R = distance / 38
        else:
            R = 1

        # 새로운 꼭지점 좌표 계산
        for coord in lmList:
            x_new = x_center + R * (coord[0] - x_center)
            y_new = height - (y_center + R * (coord[1] - y_center))
            z_new = z_center + R * (coord[2] - z_center) - distance * 5 + 500

            data.extend([x_new, y_new, z_new])


        # for lm in lmList:
        #     xCordData = lm[0]
        #     yCordData = height - lm[1]
        #     zCordData = lm[2]
        #
        #     data.extend([xCordData, yCordData, zCordData])

        # send thru port
        sock.sendto(str.encode(str(data)), serverAddressPort)

    # Hands
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
