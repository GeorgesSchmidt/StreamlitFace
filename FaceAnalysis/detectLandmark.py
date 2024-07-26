import cv2
import mediapipe as mp

class Detect:
    def __init__(self) -> None:
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        
    def detect_on_image(self, frame):
        results = self.pose.process(frame)
        h, w = frame.shape[:2]
        if results is not None:
            landmarks = results.pose_landmarks
            if landmarks is not None:
                for v in landmarks.landmark:
                    x, y = int(v.x*w), int(v.y*h)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), 2)
                return frame
                    
            
            