import cv2
import mediapipe as mp

class Detect:
    def __init__(self) -> None:
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        
    def detect_on_image(self, frame):
        results = self.pose.process(frame)
        print('results', results)