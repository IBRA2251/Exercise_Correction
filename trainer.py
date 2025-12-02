import time, math, cv2, numpy as np
from config import Model
from utils import PoseEstimator
from socket_video import SocketVideoClient

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pose = PoseEstimator()

    def run(self, out_path):
        client = SocketVideoClient()
        client.connect()

        print("Training: press 'q' to quit recording.")
        series = []
        t0 = time.time()

        while True:
            frame = client.read_frame()
            if frame is None: break
            pts = self.pose.process(frame)
            overlay = frame.copy()
            if pts:
                self.pose.draw(overlay, pts)

            now_ms = int((time.time()-t0)*1000)
            cv2.imshow("Trainer", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # TODO: add segmentation + Model saving (as in your earlier script)
        cv2.destroyAllWindows()
