from config import EXERCISE_CATALOG
from coach import LiveCoach

exercise = "bicep_curl"   # or "pushup" or "squat"
cfg = EXERCISE_CATALOG[exercise]

coach = LiveCoach(model=None, cfg=cfg)
coach.run()
