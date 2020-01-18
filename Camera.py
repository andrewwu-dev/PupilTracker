import numpy as np
import cv2
from threading import Thread, Lock

class Camera():
    def __init__(self, src, width, height):
        # Access webcam
        self.cap = cv2.VideoCapture(src)

        # Resize window
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read an initial frame
        _ , self.frame = self.cap.read()

        self.started = True

        self.locker = Lock()

    
    # Start thread execution
    def start(self) :
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        # Return this instance
        return self

    def update(self) :
        while self.started:
            _ , frame = self.cap.read()
            # Update frame
            self.locker.acquire()
            self.frame = frame
            self.locker.release()

    def read(self) :
        # Return a copy of current frame
        self.locker.acquire()
        frame = self.frame.copy() 
        # Convert to black and white
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.locker.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.cap.release()
