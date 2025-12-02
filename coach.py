import time, math, cv2
from utils import PoseEstimator
from socket_video import SocketVideoClient

class LiveCoach:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.pose = PoseEstimator()

    def run(self):
        client = SocketVideoClient()
        client.connect()

        print("Coaching session started. Press 'q' to quit.")
        while True:
            frame = client.read_frame()
            if frame is None: break
            pts = self.pose.process(frame)
            overlay = frame.copy()
            if pts:
                self.pose.draw(overlay, pts)

            cv2.imshow("Coach", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
