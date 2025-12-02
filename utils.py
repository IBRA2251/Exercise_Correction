import cv2
import numpy as np
import math
import mediapipe as mp

def angle_3pts(a, b, c):
    ba = a - b
    bc = c - b
    nba = ba / (np.linalg.norm(ba) + 1e-9)
    nbc = bc / (np.linalg.norm(bc) + 1e-9)
    cosang = np.dot(nba, nbc)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return [(lm.x * w, lm.y * h) for lm in res.pose_landmarks.landmark]

    def draw(self, frame, pts):
        if pts is None: return frame
        pairs = [(11,13),(13,15),(12,14),(14,16),(11,12),
                 (23,24),(11,23),(12,24),(23,25),(25,27),(27,29),
                 (24,26),(26,28),(28,30)]
        for i,j in pairs:
            if i < len(pts) and j < len(pts):
                cv2.line(frame, tuple(map(int,pts[i])), tuple(map(int,pts[j])), (255,255,255), 2)
        for x,y in pts:
            cv2.circle(frame, (int(x),int(y)), 3, (255,255,255), -1)
        return frame
