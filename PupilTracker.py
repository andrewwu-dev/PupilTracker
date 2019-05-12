from imutils import face_utils
import dlib
import cv2
import numpy as np


class PupilTracker():

    def __init__(self, landmark):
        # Setup detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmark)

        # Get position of right/left eye
        self.rightEyeStart, self.rightEyeEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.leftEyeStart, self.leftEyeEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

        # Setup font styles
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.fontScale = 2
        self.fontColor = (255,0,0)
        self.lineType = 2
        self.lineThickness = 1

        # Threshold to detect open eyes
        self.blinkThresh = 0.2

    def predict_blink(self, rightEye, leftEye):
        rightRatio = self.calculate_eye_ratio(rightEye)
        leftRatio = self.calculate_eye_ratio(leftEye)

        avgRatio = (rightRatio+leftRatio)/2

        if avgRatio < self.blinkThresh:
            return True

    # Based on this article:
    # https://hackaday.io/project/27552-blinktotext/log/68360-eye-blink-detection-algorithms
    def calculate_eye_ratio(self, eye):
        # Vertical point distances
        p2p6 = self.magnitude(eye[1], eye[5])
        p3p5 = self.magnitude(eye[2], eye[4])
        # Horizontal point distances
        p1p4 = self.magnitude(eye[0], eye[3])

        ratio = (p2p6 + p3p5) / (2.0 * p1p4)

        return ratio

    # Computes the magnitude of a vector
    def magnitude(self, p1, p2):
        deltaX = np.power(p1[0]-p2[0], 2)
        deltaY = np.power(p1[1]-p2[1], 2)

        return np.sqrt(deltaX + deltaY)

    def draw_pupil(self, eye, frame):
        x = min(eye[0][0], eye[1][0], eye[2][0], eye[3][0], eye[4][0])
        y = min(eye[0][1], eye[1][1], eye[2][1], eye[3][1], eye[4][1])
        w = max(eye[0][0], eye[1][0], eye[2][0], eye[3][0], eye[4][0])
        h = max(eye[0][1], eye[1][1], eye[2][1], eye[3][1], eye[4][1])

        # Extract pupil from face
        pupilFrame = frame[y:h, x:w]
        gray = cv2.cvtColor(pupilFrame, cv2.COLOR_BGR2GRAY)
        # Remove noise
        blur = cv2.GaussianBlur(gray,(11,11),0)
        
        """
        STILL TESTING

        ## CLAHE Equalization
        cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe = cl1.apply(gray)
        ## medianBlur the image to remove noise
        blur = cv2.medianBlur(clahe, 7)

        circles = cv2.HoughCircles(blur ,cv2.HOUGH_GRADIENT,1.2,20,
                            param1=10,param2=15,minRadius=7)


        if circles is not None:
            print(circles[0])
            x,y,r = circles[0]

            cv2.circle(pupilFrame,(x,y),r,(0,255,0),-1)
       
        cv2.flip(blur, image, 0);
        cv2.imshow("eye", pupilFrame)
        """

        (_, _, minLocL, _) = cv2.minMaxLoc(blur)
        cv2.circle(pupilFrame, minLocL, 3, (0, 0, 255), -1)
            
        return frame

    def detect_pupil(self, eyes):
        rightEye = eyes[0]
        leftEye = eyes[1]



    def detect(self, frame):
        # Detect faces in the gray scale image
        face = self.detector(frame, 0)

        # Exit if no face detected
        if len(face) == 0:
            return frame
        
        # Store the first face detected
        face = face[0]

        # Get right_eye, nose, mouth, etc
        facialLandmarks = self.predictor(frame, face)
        # Convert to a dictionary of specific indexes and their (x,y) coords
        facialLandmarks = face_utils.shape_to_np(facialLandmarks)

        # Extract the detected right and left eye points (Result in 2D array)
        rightEye = facialLandmarks[self.rightEyeStart:self.rightEyeEnd]
        leftEye = facialLandmarks[self.leftEyeStart:self.leftEyeEnd]

        cv2.drawContours(frame, [leftEye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEye], -1, (0, 255, 0), 1)

        # Find the probability of a blink
        didBlink = self.predict_blink(rightEye, leftEye)

        # Both eyes closed
        if didBlink:
            cv2.putText(frame,"Eyes Closed!", (20,50), self.font, self.fontScale, 
                self.fontColor, self.lineThickness, self.lineType)
        else:
            self.draw_pupil(rightEye, frame)
            self.draw_pupil(leftEye, frame)

        return frame
