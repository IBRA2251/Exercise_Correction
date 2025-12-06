import time, math, cv2, numpy as np, json, os
from config import Model
from utils import PoseEstimator
from socket_video import SocketVideoClient

# ---------- angle helper ----------
def calc_angle(a, b, c):
    """
    angle at point b formed by a-b-c in degrees (0..180)
    a, b, c are (x,y) tuples
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosv = np.dot(ba, bc) / denom
    cosv = np.clip(cosv, -1.0, 1.0)
    ang = math.degrees(math.acos(cosv))
    return ang


class Trainer:
    def __init__(self, cfg):
        """
        cfg is ExerciseConfig from config.EXERCISE_CATALOG
        expects cfg.primary_angle like "left_knee" or "left_elbow"
        and cfg.min_hold_top_ms / cfg.min_hold_bottom_ms
        """
        self.cfg = cfg
        self.pose = PoseEstimator()

    # ---------- helpers to compute primary angle ----------
    def get_primary_angle(self, pts):
        """
        pts: list of (x,y) tuples (MediaPipe ordering).
        returns angle in degrees or None if unable.
        """
        # indexes based on MediaPipe 33-landmark ordering
        # left side:
        LEFT_SHOULDER = 11
        LEFT_ELBOW = 13
        LEFT_WRIST = 15
        LEFT_HIP = 23
        LEFT_KNEE = 25
        LEFT_ANKLE = 27

        # check length
        if not pts or len(pts) < 28:
            return None

        try:
            if self.cfg.primary_angle == "left_knee":
                a = pts[LEFT_HIP]
                b = pts[LEFT_KNEE]
                c = pts[LEFT_ANKLE]
                return calc_angle(a, b, c)
            elif self.cfg.primary_angle == "left_elbow":
                a = pts[LEFT_SHOULDER]
                b = pts[LEFT_ELBOW]
                c = pts[LEFT_WRIST]
                return calc_angle(a, b, c)
            else:
                # unknown primary_angle - fallback: use centroid (not recommended)
                # compute centroid y -> map to pseudo-angle (not great)
                ys = [p[1] for p in pts if p is not None]
                if not ys:
                    return None
                norm = (max(ys) - min(ys)) + 1e-8
                centroid = sum(ys) / len(ys)
                # map centroid to 0..180 for fallback
                return 180.0 * ((centroid - min(ys)) / norm)
        except Exception:
            return None

    # ---------- main run ----------
    def run(self, out_path):
        client = SocketVideoClient()
        client.connect()

        print("Training: press 'q' to quit recording.")
        t0 = time.time()

        # we'll collect time series of angle values and timestamps
        times_s = []
        angles = []

        # store detected hold events as they occur
        top_waits = []
        bottom_waits = []

        # state machine for angle-based detection
        state = "middle"  # 'top', 'bottom', 'middle'
        enter_top_t = None
        enter_bottom_t = None

        while True:
            frame = client.read_frame()
            if frame is None:
                print("[Trainer] No frame received, stopping.")
                break

            pts = self.pose.process(frame)
            overlay = frame.copy()
            if pts:
                try:
                    self.pose.draw(overlay, pts)
                except Exception:
                    pass

            now = time.time() - t0
            times_s.append(now)

            # compute primary angle for this frame
            angle = None
            try:
                angle = self.get_primary_angle(pts)
            except Exception:
                angle = None

            angles.append(angle if angle is not None else float('nan'))

            # angle-based top/bottom detection thresholds (tunable)
            # A reasonable default: top when angle > 150 (joint near straight)
            # bottom when angle < 90 (joint bent)
            is_top = (angle is not None and angle > 150)
            is_bottom = (angle is not None and angle < 90)

            cur_t = time.time()

            # state transitions and hold measuring
            if is_top:
                if state != "top":
                    # closing bottom hold if any
                    if state == "bottom" and enter_bottom_t is not None:
                        hold = cur_t - enter_bottom_t
                        bottom_waits.append(hold)
                        enter_bottom_t = None
                    state = "top"
                    enter_top_t = cur_t
            elif is_bottom:
                if state != "bottom":
                    # closing top hold if any
                    if state == "top" and enter_top_t is not None:
                        hold = cur_t - enter_top_t
                        top_waits.append(hold)
                        enter_top_t = None
                    state = "bottom"
                    enter_bottom_t = cur_t
            else:
                # middle zone -> close any open holds
                if state == "top" and enter_top_t is not None:
                    hold = cur_t - enter_top_t
                    top_waits.append(hold)
                    enter_top_t = None
                elif state == "bottom" and enter_bottom_t is not None:
                    hold = cur_t - enter_bottom_t
                    bottom_waits.append(hold)
                    enter_bottom_t = None
                state = "middle"

            # visualization: show angle and counts
            if angle is not None:
                cv2.putText(overlay, f"Angle: {int(angle)} deg", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(overlay, f"Tops: {len(top_waits)} Bottoms: {len(bottom_waits)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow("Trainer", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Trainer] User requested quit.")
                break

        cv2.destroyAllWindows()

        # close any ongoing holds at end
        now_final = time.time()
        if state == "top" and enter_top_t is not None:
            top_waits.append(now_final - enter_top_t)
        if state == "bottom" and enter_bottom_t is not None:
            bottom_waits.append(now_final - enter_bottom_t)

        # convert waits to seconds (they already are seconds)
        top_waits_s = [float(x) for x in top_waits]
        bottom_waits_s = [float(x) for x in bottom_waits]

        # filter holds by configured minimum hold times (ms -> s)
        min_top_s = (getattr(self.cfg, "min_hold_top_ms", 150) or 150) / 1000.0
        min_bottom_s = (getattr(self.cfg, "min_hold_bottom_ms", 150) or 150) / 1000.0

        top_filtered = [t for t in top_waits_s if t >= min_top_s]
        bottom_filtered = [t for t in bottom_waits_s if t >= min_bottom_s]

        def avg(lst):
            return float(np.mean(lst)) if lst else 0.0

        summary = {
            "num_tops_recorded": len(top_waits_s),
            "num_bottoms_recorded": len(bottom_waits_s),
            "avg_top_wait_all_s": avg(top_waits_s),
            "avg_bottom_wait_all_s": avg(bottom_waits_s),
            "num_tops_filtered": len(top_filtered),
            "num_bottoms_filtered": len(bottom_filtered),
            "avg_top_wait_filtered_s": avg(top_filtered),
            "avg_bottom_wait_filtered_s": avg(bottom_filtered)
        }

        print("[Trainer] Summary:", summary)

        # --- save results per-exercise ---
        exercise_name = (getattr(self.cfg, "name", None) or "unknown_exercise").lower()
        os.makedirs("baselines", exist_ok=True)
        baseline_path = os.path.join("baselines", f"{exercise_name}_wait_baseline.json")

        baseline = {
            "exercise": exercise_name,
            "top_waits_s": top_waits_s,
            "bottom_waits_s": bottom_waits_s,
            "top_waits_filtered_s": top_filtered,
            "bottom_waits_filtered_s": bottom_filtered,
            "avg_top_wait_all_s": summary["avg_top_wait_all_s"],
            "avg_bottom_wait_all_s": summary["avg_bottom_wait_all_s"],
            "avg_top_wait_filtered_s": summary["avg_top_wait_filtered_s"],
            "avg_bottom_wait_filtered_s": summary["avg_bottom_wait_filtered_s"],
            "min_hold_top_ms": getattr(self.cfg, "min_hold_top_ms", None),
            "min_hold_bottom_ms": getattr(self.cfg, "min_hold_bottom_ms", None),
            "saved_at": int(time.time())
        }

        # save canonical file and timestamped history
        try:
            with open(baseline_path, "w") as f:
                json.dump(baseline, f, indent=2)
            print(f"[Trainer] Saved baseline to {baseline_path}")
        except Exception as e:
            print("[Trainer] Error saving baseline:", e)

        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            hist_path = os.path.join("baselines", f"{exercise_name}_wait_baseline_{ts}.json")
            with open(hist_path, "w") as hf:
                json.dump(baseline, hf, indent=2)
            print(f"[Trainer] Saved historical baseline to {hist_path}")
        except Exception as e:
            print("[Trainer] Error saving historical baseline:", e)

        # optional: attempt to call Model.train / save
        try:
            mdl = Model(self.cfg)
            if hasattr(mdl, "train"):
                try:
                    mdl.train({
                        "times_s": times_s,
                        "angles": angles,
                        "top_waits_s": top_waits_s,
                        "bottom_waits_s": bottom_waits_s
                    }, out_path)
                except Exception:
                    # ignore if signature mismatch
                    pass
            if hasattr(mdl, "save"):
                try:
                    mdl.save(os.path.join("baselines", f"{exercise_name}.model"))
                except Exception:
                    pass
        except Exception:
            pass

        print("[Trainer] Finished.")
