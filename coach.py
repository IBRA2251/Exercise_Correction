import time, math, cv2, json, os
import numpy as np
from utils import PoseEstimator
from socket_video import SocketVideoClient


#   ANGLE CALCULATION
def calc_angle(a, b, c):
    """Calculates angle at point b between a-b-c (0–180 degrees)."""
    a, b, c = map(lambda p: np.array(p, dtype=float), (a, b, c))

    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosv = np.dot(ba, bc) / denom
    angle = np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))

    return float(angle)


#   COACH CLASS
class LiveCoach:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.pose = PoseEstimator()

        # LOAD BASELINE
        exercise_name = self.cfg.name.lower()
        baseline_file = f"baselines/{exercise_name}_wait_baseline.json"

        if not os.path.exists(baseline_file):
            raise FileNotFoundError(f"[Coach] Baseline not found: {baseline_file}. Run trainer first.")

        with open(baseline_file, "r") as f:
            baseline = json.load(f)

        # SMART KEYS
        self.top_baseline = (
            baseline.get("avg_top_wait_filtered_s") or
            baseline.get("avg_top_wait_all_s") or
            baseline.get("avg_top_wait_s") or
            1.0
        )
        self.bottom_baseline = (
            baseline.get("avg_bottom_wait_filtered_s") or
            baseline.get("avg_bottom_wait_all_s") or
            baseline.get("avg_bottom_wait_s") or
            1.0
        )

        print(f"[Coach] Loaded baseline for {exercise_name}:")
        print("   Top baseline:", self.top_baseline)
        print("   Bottom baseline:", self.bottom_baseline)

        # Precision thresholds
        self.TOP_THRESHOLD = 140       # was 150 – more forgiving
        self.BOTTOM_THRESHOLD = 100    # was 90 – more forgiving
        self.TOL = 0.30                # ±30% hold tolerance

        # State machine
        self.state = "top"       # assume user starts standing/extended
        self.last_transition_time = None
        self.enter_top_t = None
        self.enter_bottom_t = None

        # Rep tracking
        self.rep_count = 0
        self.rep_phase = "down"   # "down" → going bottom ; "up" → going top
        self.rep_start_angle = None

        # Hold tracking for one rep
        self.top_hold = 0
        self.bottom_hold = 0

        # Feedback
        self.last_feedback = ""


    #   COMPUTE PRIMARY ANGLE
    def get_angle(self, pts):
        L_SHOULDER = 11; L_ELBOW = 13; L_WRIST = 15
        L_HIP = 23; L_KNEE = 25; L_ANKLE = 27

        if self.cfg.primary_angle == "left_knee":
            return calc_angle(pts[L_HIP], pts[L_KNEE], pts[L_ANKLE])

        if self.cfg.primary_angle == "left_elbow":
            return calc_angle(pts[L_SHOULDER], pts[L_ELBOW], pts[L_WRIST])

        return None


    #   EVALUATE REP AND GIVE FEEDBACK
    def evaluate_rep(self, rom, top_hold, bottom_hold, tempo):
        feedback = []
        score = 100

        # ---------- ROM ----------
        if rom < self.cfg.min_rom:
            feedback.append("ROM too shallow")
            score -= 35
        else:
            feedback.append("Good ROM")

        # ---------- TOP HOLD ----------
        if top_hold < self.top_baseline * 0.7:
            feedback.append("Top hold too short")
            score -= 15
        elif top_hold > self.top_baseline * 1.3:
            feedback.append("Top hold too long")
            score -= 10
        else:
            feedback.append("Good top hold")

        # ---------- BOTTOM HOLD ----------
        if bottom_hold < self.bottom_baseline * 0.7:
            feedback.append("Bottom hold too short")
            score -= 15
        elif bottom_hold > self.bottom_baseline * 1.3:
            feedback.append("Bottom hold too long")
            score -= 10
        else:
            feedback.append("Good bottom hold")

        # ---------- TEMPO ----------
        if tempo < 0.6:
            feedback.append("Too fast")
            score -= 20
        elif tempo > 2.0:
            feedback.append("Too slow")
            score -= 10
        else:
            feedback.append("Good tempo")

        # Final judgment
        if score >= 85:
            summary = "Excellent rep!"
        elif score >= 70:
            summary = "Good rep."
        else:
            summary = "Poor rep — needs improvement."

        self.last_feedback = summary + " | " + ", ".join(feedback)


    #   MAIN COACH LOOP
    def run(self):
        client = SocketVideoClient()
        client.connect()

        print("Coaching started. Press 'q' to exit.")

        while True:
            frame = client.read_frame()
            if frame is None:
                break

            pts = self.pose.process(frame)
            overlay = frame.copy()
            if pts:
                self.pose.draw(overlay, pts)

            if not pts:
                cv2.imshow("Coach", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ---------------- ANGLE ----------------
            angle = self.get_angle(pts)
            if angle is None:
                continue

            t = time.time()

            is_top = angle > self.TOP_THRESHOLD
            is_bottom = angle < self.BOTTOM_THRESHOLD


            # ---------------- ON TOP ----------------
            if is_top:
                if self.state != "top":
                    # leaving bottom → measure bottom hold
                    if self.state == "bottom" and self.enter_bottom_t:
                        self.bottom_hold = t - self.enter_bottom_t

                    # if we were rising up from bottom → rep complete
                    if self.rep_phase == "up":
                        self.rep_count += 1
                        rom = abs(self.rep_start_angle - angle)
                        tempo = t - (self.last_transition_time or t)

                        self.evaluate_rep(
                            rom=rom,
                            top_hold=self.top_hold,
                            bottom_hold=self.bottom_hold,
                            tempo=tempo
                        )

                    # reset for next rep
                    self.rep_phase = "down"
                    self.rep_start_angle = angle
                    self.state = "top"
                    self.enter_top_t = t
                    self.last_transition_time = t

            # ---------------- ON BOTTOM ----------------
            elif is_bottom:
                if self.state != "bottom":
                    # leaving top → measure top hold
                    if self.state == "top" and self.enter_top_t:
                        self.top_hold = t - self.enter_top_t

                    self.state = "bottom"
                    self.enter_bottom_t = t
                    self.last_transition_time = t
                    self.rep_phase = "up"

            # ---------------- MIDDLE ----------------
            else:
                self.state = "middle"

            #   DISPLAY OVERLAYS
            cv2.putText(overlay, f"Angle: {int(angle)}°", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            cv2.putText(overlay, f"Reps: {self.rep_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

            if self.last_feedback:
                cv2.putText(overlay, self.last_feedback, (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Coach", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
