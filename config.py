from dataclasses import dataclass, asdict
import json


@dataclass
class ExerciseConfig:
    name: str
    primary_angle: str
    direction: str
    min_rom: float
    min_hold_top_ms: int = 150
    min_hold_bottom_ms: int = 150


EXERCISE_CATALOG = {
    "squat": ExerciseConfig("squat", "left_knee", "decrease_then_increase", 35, 200, 200),
    "pushup": ExerciseConfig("pushup", "left_elbow", "decrease_then_increase", 30, 150, 150),
    "bicep_curl": ExerciseConfig("bicep_curl", "left_elbow", "decrease_then_increase", 40, 150, 150),
}


@dataclass
class Model:
    exercise: str
    primary_angle: str
    rom_mean: float
    rom_std: float
    tempo_mean_ms: float
    tempo_std_ms: float
    hold_top_mean_ms: float
    hold_top_std_ms: float
    hold_bottom_mean_ms: float
    hold_bottom_std_ms: float

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s):
        return Model(**json.loads(s))
