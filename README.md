# PupilTracker
Python script that tracks pupils by first extracting an eye frame and drawing a dot at the darkest point. 
Also detects when eyes are closed.

![pupilTracker.gif](pupilTracker.gif)

## Libraries Used
- OpenCV
- Dlib
- Imutils

## Sources
- [68-point landmark model](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat)

- [Blink detection](https://hackaday.io/project/27552-blinktotext/log/68360-eye-blink-detection-algorithms)

- [Pupil correction](http://marcopellin.weebly.com/uploads/3/7/9/5/37955055/center_of_pupil_detection_m._pellin.pdf)

