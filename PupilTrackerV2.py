from imutils import face_utils
import dlib
import cv2
import numpy as np

# Setup Model
# This a pretrained model obtained from
# https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
landmark = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(leftIndexStart, rightIndexEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightIndexStart, rightIndexEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Add a little more accuracy, still need to play around
xPadding = 5 #5 default
yPadding = 3 #3 default
# Threshold to detect open eyes, still need to play around with
blinkThresh = 0.2

font = cv2.FONT_HERSHEY_PLAIN
bottomLeftCornerOfText = (10,400)
fontScale = 1
fontColor = (255,0,0)
lineType = 2

xScreen = 640
yScreen = 480

# Function to calculate eye aspect ratio
# Used for detecting closed eyes (might need might not)
# Refer to link below for formula + explanation
# https://hackaday.io/project/27552-blinktotext/log/68360-eye-blink-detection-algorithms
def find_eye_ratio(eye):
    # Find distances between vertical eye landmark points
    print(eye[1])
    p2p6 = magnitude(eye[1],eye[5])
    p3p5 = magnitude(eye[2],eye[4])
    
    # Find distance between horizontal eye landmark points
    p1p4 = magnitude(eye[0],eye[3])
    
    # compute the eye aspect ratio (ear)
    ear = (p2p6 + p3p5) / (2.0 * p1p4)
    
    # return the eye aspect ratio
    return ear

def magnitude(p1, p2):
    deltaX = np.power(p1[0]-p2[0], 2)
    deltaY = np.power(p1[1]-p2[1], 2)
    
    return np.sqrt(deltaX + deltaY)

def find_center_pupil(x1,x2,y1,y2):
    center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
    
    return center

def convert_to_screen_coords(x,y,rectWidth,rectHeight):
    xMultiplier = float(rectWidth/xScreen)
    xNew = float(x/xMultiplier)
    
    yMultiplier = float(rectHeight/yScreen)
    yNew = float(y/yMultiplier)
    
    return [xNew,yNew]

def detect_gaze(leftPupil, rightPupil, leftPupilOrigin, rightPupilOrigin):
    # Get the average the of actual location of the pupils
    pupil_avg = [
                 (leftPupil[0] + rightPupil[0]) / 2,
                 (leftPupil[1] + rightPupil[1]) / 2
                 ]
        
                 # Get the average of origin points
                 pupil_origin_avg = [
                                     (leftPupilOrigin[0] + rightPupilOrigin[0]) / 2,
                                     (leftPupilOrigin[1] + rightPupilOrigin[1]) / 2
                                     ]
                 
                 # Index 0 is x value, index 1 is y value
                 
                 threshold = 0.7
                 
                 # Centered
                 if pupil_avg[0] / pupil_origin_avg[0] > threshold and pupil_avg[1] / pupil_origin_avg[1] > threshold:
                     cv2.putText(image, 'CENTER', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                 else:
                     if pupil_avg[0] > pupil_origin_avg[0]:
                         cv2.putText(image, 'LEFT', (50,400), font, fontScale, fontColor, lineType)
                             else:
                                 cv2.putText(image, 'RIGHT', (50,400), font, fontScale, fontColor, lineType)
                                     
                                     if pupil_avg[1] < pupil_origin_avg[1]:
                                         cv2.putText(image, 'TOP', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                                             else:
                                                 cv2.putText(image, 'BOTTOM', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

if __name__ == "__main__":
    # Setup Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, xScreen)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, yScreen)
    
    while True:
        # Get image captured by webcam
        _, image = cap.read()
        
        # Convert screen capture to black-white image
        # (Make predictions more accurate? Need more research)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # For more accurate results with cv2.maxMinLoc use a blur
        # Default region (11,11), need more testing
        blurred = cv2.GaussianBlur(gray,(11,11),0)
        
        # Detect faces in the gray scale image
        face = detector(gray, 0)[0]
        # Get face landmarks i.e. right_eye, nose, mouth, etc
        # finding location of darker pixel inside eye region
        # eyeRect is black-white so different eye colors dont matter
        facialLandmarks = predictor(gray, face)
        # Convert to a dictionary of specific indexes and their (x,y) coords
        facial_landmarks = face_utils.shape_to_np(facial_landmarks)
        
        # Extract the detected left and right eye points (Result in 2D array)
        leftEye = facial_landmarks[LSTART:LEND]
        rightEye = facial_landmarks[RSTART:REND]
        
        # Compute the aspect ratio for both eyes (EAR stand for eye aspect ratio)
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # average the eye aspect ratio together for both eyes
        ear_avg = (leftEAR + rightEAR) / 2.0
        
        # Location of left bounding box
        xLeft = min(leftEye[0][0],leftEye[1][0],leftEye[2][0],leftEye[3][0],leftEye[4][0])
        yLeft = min(leftEye[0][1],leftEye[1][1],leftEye[2][1],leftEye[3][1],leftEye[4][1])
        widthL = max(leftEye[0][0],leftEye[1][0],leftEye[2][0],leftEye[3][0],leftEye[4][0])
        heightL = max(leftEye[0][1],leftEye[1][1],leftEye[2][1],leftEye[3][1],leftEye[4][1])
        
        # Location of Right bounding box
        xRight = min(rightEye[0][0], rightEye[1][0], rightEye[2][0], rightEye[3][0], rightEye[4][0])
        yRight = min(rightEye[0][1], rightEye[1][1], rightEye[2][1], rightEye[3][1], rightEye[4][1])
        widthR = max(rightEye[0][0], rightEye[1][0], rightEye[2][0], rightEye[3][0], rightEye[4][0])
        heightR = max(rightEye[0][1], rightEye[1][1], rightEye[2][1], rightEye[3][1], rightEye[4][1])
        
        # Calculate eye center
        leftPupilOrigin = find_center_pupil(xLeft, widthL, yLeft, heightL)
        rightPupilOrigin = find_center_pupil(xRight, widthR, yRight, heightR)
        
        #print("x: ", xLeft, " y: ", yLeft, " width: ", widthL, " height: ", heightL)
        
        # draw rectangle around eyes
        cv2.rectangle(image, (xLeft + PADDING_X, yLeft), \
                      (widthL - PADDING_X, heightL - PADDING_Y), (0, 255, 0), 1)
                      cv2.rectangle(image, (xRight + PADDING_X, yRight), \
                                    (widthR - PADDING_X, heightR - PADDING_Y), (0, 255, 0), 1)
                      
                      # Extracting region of left eye for further process
                      # Essentially focus frame with eye
                      # Extract one black-white frame to process and one colored frame to draw on
                      leftImage = blurred[yLeft:heightL, xLeft + PADDING_X:widthL - PADDING_X]
                      leftColoredImage = image[yLeft:heightL, xLeft + PADDING_X:widthL - PADDING_X]
                      
                      # Extracting region of right eye for further process
                      rightImage = blurred[yRight:heightR - PADDING_Y, xRight + PADDING_X:widthR - PADDING_X]
                      rightColoredImage = image[yRight:heightR - PADDING_Y, xRight + PADDING_X:widthR - PADDING_X]
                      
                      # Verify that eyes are not closed and draw the pupils if true
                      if ear_avg >= EAR_THRESH:
                          # Draws onto colored image so we can see it in the window
                          # cv2.minMaxLoc finds the brightest(maxLoc) or darkest(minLoc) spot within region
                          # Returns coords of the pixel which will be used as the center coord of pupil
                          (_, _, minLocL, _) = cv2.minMaxLoc(leftImage)
                          cv2.circle(leftColoredImage, minLocL, 5, (0, 0, 255), -1)
                          
                          (_, _, minLocR, _) = cv2.minMaxLoc(rightImage)
                          cv2.circle(rightColoredImage, minLocR, 5, (0, 0, 255), -1)
                          
                          # Calculate average bounding box size of eye
                          rectWidth = ((widthL + widthR) / 2.0) - ((xLeft + xRight) / 2.0)
                          rectHeight = ((heightL + heightR) / 2.0) - ((yLeft + yRight) / 2.0)
                          
                          # Screen coords of predicted pupil
                          xLeftPupil = convert_to_screen_coords(minLocL[0], minLocL[1], rectWidth, rectHeight)
                          yRightPupil = convert_to_screen_coords(minLocR[0], minLocR[1], rectWidth, rectHeight)
                          
                              detect_gaze(xLeftPupil, yRightPupil, leftPupilOrigin, rightPupilOrigin)
                          
                          # print("left eye: ", minLocL, " right eye: ", minLocR)
                          
                          
                          
                          
                          
                          # show camera feed
                          cv2.imshow("Overall", image)
                              
                              # press q to quit
                              if cv2.waitKey(1) & 0xFF == ord('q'):
                                  break

cv2.destroyAllWindows()
cap.release()


        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 2, 20,
                                   param1=20, param2=20, minRadius=10, maxRadius=15)
        
        print("circles: ", circles)
        
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x,y,r = circles[0]
            cv2.circle(pupilFrame, (x, y), r, (0, 255, 0), -1)
            #cv2.rectangle(pupilFrame, (x - 5, y - 5),(x + 5, y + 5), (0, 128, 255), -1)