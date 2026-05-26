/**
 * AdminTrainer — exact browser port of trainer.py
 * Records the admin's exercise, computes baseline model stats,
 * and produces model.json + motion.json for the AI coach.
 */

// All 6 joints tracked — mirrors trainer.py JOINTS dict
const TRAINER_JOINTS = {
    left_knee:   [23, 25, 27],
    right_knee:  [24, 26, 28],
    left_elbow:  [11, 13, 15],
    right_elbow: [12, 14, 16],
    left_hip:    [11, 23, 25],
    right_hip:   [12, 24, 26],
};

// Primary joint per exercise (for adaptive rep detection)
const TRAINER_PRIMARY_JOINT = {
    squat:      'left_knee',
    pushup:     'left_elbow',
    bicep_curl: 'left_elbow',
    pullup:     'left_elbow',
    lunge:      'left_knee',
    boxing:     'right_elbow',
    yoga:       'left_knee',
    stretching: 'left_knee',
};

// Maps each primary joint to its mirror on the other side
// Used to automatically select the right-arm primary joint during dual-arm training
const TRAINER_JOINT_OPPOSITE = {
    left_elbow:  'right_elbow',
    right_elbow: 'left_elbow',
    left_knee:   'right_knee',
    right_knee:  'left_knee',
};

const MIN_ANGLE_RANGE = 15;   // mirrors trainer.py MIN_ANGLE_RANGE
const MIN_REP_GAP_S   = 0.5;  // seconds — mirrors trainer.py MIN_REP_GAP
const MAX_MOTION_FRAMES = 300; // subsample cap to keep file size reasonable

// ── Math helpers ──────────────────────────────────────────────────────────────
function trainerAngle(a, b, c) {
    const rad = Math.atan2(c[1]-b[1], c[0]-b[0]) - Math.atan2(a[1]-b[1], a[0]-b[0]);
    let deg = Math.abs(rad * 180 / Math.PI);
    if (deg > 180) deg = 360 - deg;
    return deg;
}

function arrStats(arr) {
    if (!arr.length) return { min:0, max:0, avg:0, std:0 };
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const avg = arr.reduce((s,v)=>s+v, 0) / arr.length;
    const std = Math.sqrt(arr.reduce((s,v)=>s+(v-avg)**2, 0) / arr.length);
    return { min, max, avg, std };
}

function arrSpeedStats(arr) {
    if (!arr.length) return { speed_avg:0, speed_std:0 };
    const avg = arr.reduce((s,v)=>s+v, 0) / arr.length;
    const std = Math.sqrt(arr.reduce((s,v)=>s+(v-avg)**2, 0) / arr.length);
    return { speed_avg: avg, speed_std: std };
}

// mirrors numpy.percentile (linear interpolation)
function percentile(arr, p) {
    if (!arr.length) return 0;
    const sorted = [...arr].sort((a,b) => a-b);
    const idx = (p / 100) * (sorted.length - 1);
    const lo  = Math.floor(idx);
    const hi  = Math.ceil(idx);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

// ── AdminTrainer class ────────────────────────────────────────────────────────
class AdminTrainer {
    constructor() {
        this.poseEngine   = null;
        this.isRecording  = false;

        // Collected data
        this.motionFrames = [];          // [[x,y], ...] per frame for all 33 lm
        this.jointAngles  = {};          // { joint: [angle, ...] }
        this.jointSpeeds  = {};          // { joint: [speed, ...] }
        this.prevAngles   = {};          // last frame angle per joint
        this.prevTs       = 0;           // performance.now() of previous frame
        this.centroidY    = [];          // body centroid Y per frame

        // Adaptive rep detection (mirrors trainer.py)
        this.repCount     = 0;
        this.repState     = 'top';
        this.topAngle     = null;        // initialised on first frame (mirrors trainer.py: top_angle = None)
        this.bottomAngle  = null;        // initialised on first frame (mirrors trainer.py: bottom_angle = None)
        this.lastRepTs    = 0;           // seconds

        // Config (set by setup())
        this.exerciseId   = null;
        this.exerciseName = null;
        this.exerciseType = null;
        this.targetReps   = 10;
        this.primaryJoint = null;

        // Output
        this.model  = null;
        this.motion = null;

        // UI callbacks
        this.onRepCount    = null;  // (count) => void
        this.onPhaseChange = null;  // ('TOP' | 'BOTTOM') => void
        this.onAngle       = null;  // (deg) => void
        this.onComplete    = null;  // (model, motion) => void
        this.onError       = null;  // (msg) => void
    }

    // ── Setup ─────────────────────────────────────────────────────────────────
    // armSide: 'left' (default) | 'right'
    // When 'right', the primary joint is automatically swapped to its mirror.
    setup(exercise, targetReps, armSide = 'left') {
        this.exerciseId   = exercise.id;
        this.exerciseName = exercise.name;
        this.exerciseType = exercise.type;
        this.targetReps   = Math.max(3, targetReps);
        this.armSide      = armSide;

        const defaultPrimary = TRAINER_PRIMARY_JOINT[exercise.id] || 'left_knee';
        this.primaryJoint = (armSide === 'right' && TRAINER_JOINT_OPPOSITE[defaultPrimary])
            ? TRAINER_JOINT_OPPOSITE[defaultPrimary]
            : defaultPrimary;

        Object.keys(TRAINER_JOINTS).forEach(j => {
            this.jointAngles[j] = [];
            this.jointSpeeds[j] = [];
            this.prevAngles[j]  = null;
        });
    }

    // ── Camera init ───────────────────────────────────────────────────────────
    async startCamera(videoEl, canvasEl) {
        this.poseEngine = new PoseEngine();

        // Highlight primary joint on canvas
        const jDef = TRAINER_JOINTS[this.primaryJoint];
        if (jDef) this.poseEngine.primaryJointIdx = jDef[1];

        this.poseEngine.onResults = (landmarks, w, h) => {
            if (!landmarks) return;
            if (this.isRecording) this._processFrame(landmarks, w, h);
        };

        await this.poseEngine.init(videoEl, canvasEl);
    }

    // ── Start / Stop ─────────────────────────────────────────────────────────
    startRecording() {
        // Reset all data
        this.motionFrames = [];
        Object.keys(TRAINER_JOINTS).forEach(j => {
            this.jointAngles[j] = [];
            this.jointSpeeds[j] = [];
            this.prevAngles[j]  = null;
        });
        this.centroidY   = [];
        this.repCount    = 0;
        this.repState    = 'top';
        this.topAngle    = null;   // seeded from first frame (mirrors trainer.py: top_angle = None)
        this.bottomAngle = null;   // seeded from first frame (mirrors trainer.py: bottom_angle = None)
        this.lastRepTs   = 0;
        this.prevTs      = performance.now();

        this.isRecording = true;
    }

    stopRecording() {
        if (!this.isRecording) return;
        this.isRecording = false;
        this._computeResults();
    }

    stopCamera() {
        this.isRecording = false;
        if (this.poseEngine) { this.poseEngine.stop(); this.poseEngine = null; }
    }

    // ── Per-frame processing ──────────────────────────────────────────────────
    _processFrame(landmarks, w, h) {
        const nowTs  = performance.now();
        const dt     = (nowTs - this.prevTs) / 1000;   // seconds
        const fps    = dt > 0 ? 1 / dt : 30;
        this.prevTs  = nowTs;
        const nowS   = nowTs / 1000;

        // ── Motion data: normalized [x,y] per landmark ─────────────────────
        this.motionFrames.push(landmarks.map(lm => [lm.x, lm.y]));

        // ── Body centroid Y (for hold time analysis) ────────────────────────
        const bodySlice = landmarks.slice(11, 29);
        const cy = bodySlice.reduce((s,lm)=>s+lm.y, 0) / bodySlice.length;
        this.centroidY.push(cy);

        // ── All joint angles + speeds ───────────────────────────────────────
        for (const [jName, [ia, ib, ic]] of Object.entries(TRAINER_JOINTS)) {
            if (ic >= landmarks.length) continue;
            const a = [landmarks[ia].x, landmarks[ia].y];
            const b = [landmarks[ib].x, landmarks[ib].y];
            const c = [landmarks[ic].x, landmarks[ic].y];
            const angle = trainerAngle(a, b, c);
            this.jointAngles[jName].push(angle);

            if (this.prevAngles[jName] !== null) {
                this.jointSpeeds[jName].push(Math.abs(angle - this.prevAngles[jName]) * fps);
            }
            this.prevAngles[jName] = angle;
        }

        // ── Adaptive rep detection on primary joint ─────────────────────────
        const pjDef = TRAINER_JOINTS[this.primaryJoint];
        if (!pjDef) return;
        const [pa, pb, pc] = pjDef;
        if (pc >= landmarks.length) return;

        const pAngle = trainerAngle(
            [landmarks[pa].x, landmarks[pa].y],
            [landmarks[pb].x, landmarks[pb].y],
            [landmarks[pc].x, landmarks[pc].y]
        );

        // Adaptive tracking (mirrors trainer.py lines 333-348)
        // Seed both values from the very first observed angle (same as Python: top_angle = None → set on first frame)
        if (this.topAngle === null) {
            this.topAngle    = pAngle;
            this.bottomAngle = pAngle;
        } else {
            this.topAngle    = Math.max(this.topAngle,    pAngle);
            this.bottomAngle = Math.min(this.bottomAngle, pAngle);
        }

        // Update canvas angle label
        if (this.poseEngine) this.poseEngine.currentAngle = pAngle;
        if (this.onAngle) this.onAngle(Math.round(pAngle));

        const rom = this.topAngle - this.bottomAngle;

        if (rom > MIN_ANGLE_RANGE) {
            if (pAngle < this.bottomAngle + 10 && this.repState === 'top') {
                this.repState = 'bottom';
                if (this.onPhaseChange) this.onPhaseChange('BOTTOM');

            } else if (pAngle > this.topAngle - 10 && this.repState === 'bottom') {
                if (nowS - this.lastRepTs > MIN_REP_GAP_S) {
                    this.repCount++;
                    this.repState  = 'top';
                    this.lastRepTs = nowS;
                    if (this.onRepCount) this.onRepCount(this.repCount);
                    if (this.onPhaseChange) this.onPhaseChange('TOP');

                    // Auto-stop when target reached
                    if (this.repCount >= this.targetReps) {
                        this.isRecording = false;
                        this._computeResults();
                    }
                }
            }
        }
    }

    // ── Compute baseline model + motion ───────────────────────────────────────
    _computeResults() {
        if (this.repCount < 1) {
            if (this.onError) this.onError('No reps detected. Try again.');
            return;
        }

        // ── Joint stats (mirrors trainer.py lines 437-445) ─────────────────
        const joints = {};
        for (const jName of Object.keys(TRAINER_JOINTS)) {
            const angles = this.jointAngles[jName];
            const speeds = this.jointSpeeds[jName];
            if (!angles.length) continue;

            const as = arrStats(angles);
            const ss = arrSpeedStats(speeds);

            joints[jName] = {
                min:       parseFloat(as.min.toFixed(2)),
                max:       parseFloat(as.max.toFixed(2)),
                avg:       parseFloat(as.avg.toFixed(2)),
                std:       parseFloat(as.std.toFixed(2)),
                speed_avg: parseFloat(ss.speed_avg.toFixed(2)),
                speed_std: parseFloat(ss.speed_std.toFixed(2)),
            };
        }

        // ── Hold times (mirrors trainer.py lines 406-408) ──────────────────
        let holdTopAvg = 0, holdBottomAvg = 0;
        if (this.centroidY.length) {
            const topLevel    = percentile(this.centroidY, 10);
            const bottomLevel = percentile(this.centroidY, 90);
            const topFrames    = this.centroidY.filter(y => y < topLevel).length;
            const bottomFrames = this.centroidY.filter(y => y > bottomLevel).length;
            holdTopAvg    = parseFloat((topFrames    / 30).toFixed(2));
            holdBottomAvg = parseFloat((bottomFrames / 30).toFixed(2));
        }

        // ── Model JSON ─────────────────────────────────────────────────────
        this.model = {
            exercise:          this.exerciseId,
            type:              this.exerciseType,
            reps_recorded:     this.repCount,
            arm_side:          this.armSide,   // 'left' | 'right'
            joints,
            hold_top_avg_s:    holdTopAvg,
            hold_bottom_avg_s: holdBottomAvg,
        };

        // ── Motion JSON — subsample to MAX_MOTION_FRAMES ───────────────────
        const total = this.motionFrames.length;
        const step  = total > MAX_MOTION_FRAMES ? Math.ceil(total / MAX_MOTION_FRAMES) : 1;
        this.motion = this.motionFrames.filter((_, i) => i % step === 0);

        if (this.onComplete) this.onComplete(this.model, this.motion);
    }
}

window.AdminTrainer          = AdminTrainer;
window.TRAINER_JOINTS        = TRAINER_JOINTS;
window.TRAINER_PRIMARY_JOINT = TRAINER_PRIMARY_JOINT;
window.TRAINER_JOINT_OPPOSITE= TRAINER_JOINT_OPPOSITE;
