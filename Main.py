import numpy as np
import cv2
from Camera import Camera
from PupilTracker import PupilTracker

# Setup PupilTracker   
print("Setting up detector and predictor...")
# Location of model file
# This a pretrained model obtained from
# https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
tracker = PupilTracker("shape_predictor_68_face_landmarks.dat")

# Setup Camera
print("Setting up camera...")
# src=webcam(0), width=640, height=480
cam = Camera(0,640,480).start()


while True:
    frame = cam.read()

    frame = tracker.detect(frame)

    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows