/**
 * Exercise FSM Engine — exact JavaScript port of fsm.py + coach.py feedback logic
 *
 * Classes: SingleExerciseFSM, BoxingFSM, YogaFSM (mirrors fsm.py)
 * evaluateRep(): mirrors coach.py LiveCoach.evaluate_rep() with full feedback tables
 */

// ─── JOINTS mapping (identical to coach.py) ──────────────────────────────────
const JOINTS = {
    left_elbow:    [11, 13, 15],
    right_elbow:   [12, 14, 16],
    left_knee:     [23, 25, 27],
    right_knee:    [24, 26, 28],
    left_shoulder: [13, 11, 23],
    right_shoulder:[14, 12, 24],
};

const OPPOSITE_JOINT = {
    left_elbow:    'right_elbow',
    right_elbow:   'left_elbow',
    left_knee:     'right_knee',
    right_knee:    'left_knee',
    left_shoulder: 'right_shoulder',
    right_shoulder:'left_shoulder',
};

// ─── 3-point angle (same as coach.py calculate_angle) ────────────────────────
function calculateAngle(a, b, c) {
    const ba  = [a[0] - b[0], a[1] - b[1]];
    const bc  = [c[0] - b[0], c[1] - b[1]];
    const dot = ba[0]*bc[0] + ba[1]*bc[1];
    const mag = (Math.sqrt(ba[0]**2+ba[1]**2)*Math.sqrt(bc[0]**2+bc[1]**2)) + 1e-9;
    return (Math.acos(Math.max(-1, Math.min(1, dot/mag)))) * 180 / Math.PI;
}

// ─── Base FSM ─────────────────────────────────────────────────────────────────
class BaseFSM {
    constructor() {
        this.state          = null;
        this.prevState      = null;
        this.stateEnterTime = performance.now();
    }
    transition(newState) {
        if (newState !== this.state) {
            this.prevState      = this.state;
            this.state          = newState;
            this.stateEnterTime = performance.now();
        }
    }
    timeInState() {
        return (performance.now() - this.stateEnterTime) / 1000;
    }
}

// ─── SingleExerciseFSM (mirrors fsm.py) ───────────────────────────────────────
class SingleExerciseFSM extends BaseFSM {
    constructor(minAngle, maxAngle, minRom) {
        super();
        this.minAngle = minAngle;
        this.maxAngle = maxAngle;
        this.minRom   = minRom;
        this.state    = 'top';
    }
    update(angle) {
        const rom = this.maxAngle - this.minAngle;
        if (rom < this.minRom) return this.state;

        // Relaxed thresholds (±15° instead of ±5°) so normal-speed movement is recognised
        if      (this.state === 'top'    && angle <  this.maxAngle - 10) this.transition('down');
        else if (this.state === 'down'   && angle <= this.minAngle + 15) this.transition('bottom');
        else if (this.state === 'bottom' && angle >  this.minAngle + 10) this.transition('up');
        else if (this.state === 'up'     && angle >= this.maxAngle - 15) this.transition('top');
        return this.state;
    }
}

// ─── BoxingFSM (mirrors fsm.py) ───────────────────────────────────────────────
class BoxingFSM extends BaseFSM {
    constructor() {
        super();
        this.state         = 'guard';
        this.PUNCH_SPEED   = 120;
        this.RECOVER_SPEED = 40;
        this.MIN_PUNCH_TIME= 0.15;
    }
    update(speed) {
        if      (this.state === 'guard'   && speed > this.PUNCH_SPEED)                    this.transition('punch');
        else if (this.state === 'punch'   && this.timeInState() > this.MIN_PUNCH_TIME)    this.transition('recover');
        else if (this.state === 'recover' && speed < this.RECOVER_SPEED)                  this.transition('guard');
        return this.state;
    }
}

// ─── YogaFSM (mirrors fsm.py) ─────────────────────────────────────────────────
class YogaFSM extends BaseFSM {
    constructor() {
        super();
        this.state        = 'entering';
        this.STABLE_STD   = 3.0;
        this.UNSTABLE_STD = 6.0;
    }
    update(angleStd) {
        if      (this.state === 'entering' && angleStd < this.STABLE_STD)   this.transition('hold');
        else if (this.state === 'hold'     && angleStd > this.UNSTABLE_STD) this.transition('exiting');
        else if (this.state === 'exiting'  && angleStd < this.STABLE_STD)   this.transition('hold');
        return this.state;
    }
}

// ─── FSM Factory ─────────────────────────────────────────────────────────────
function createFSM(cfg, model) {
    if (cfg.type === 'boxing') return new BoxingFSM();
    if (cfg.type === 'yoga')   return new YogaFSM();
    const stats = model?.joints?.[cfg.primary_angle];
    if (stats) return new SingleExerciseFSM(stats.min, stats.max, cfg.min_rom);
    return new SingleExerciseFSM(30, 160, cfg.min_rom || 30);
}

// ════════════════════════════════════════════════════════════════════════════════
//  ExerciseSession — rep counting + per-rep evaluation
//  Mirrors coach.py LiveCoach.run() + evaluate_rep()
// ════════════════════════════════════════════════════════════════════════════════
class ExerciseSession {
    constructor(cfg, model) {
        this.config = cfg;
        this.model  = model;
        this.fsm    = createFSM(cfg, model);

        this.repCount   = 0;
        this.repState   = 'rest';   // mirrors coach.py rep_state
        this.lastRepTime= 0;
        this.MIN_REP_GAP= 0.5;     // seconds between reps (debounce)

        this.prevAngle          = null;
        this.speedBuffer        = [];
        this.FPS                = 30;
        this._lastFrameTime     = null;

        // Per-rep tracking (mirrors coach.py rep_min_angle, rep_max_angle, rep_speed_samples)
        this.repMinAngle    = null;
        this.repMaxAngle    = null;
        this.repSpeedSamples= [];

        // Quality
        this.qualityScores  = [];
        this.currentQuality = 0;

        // Multi-stage tracking
        this.prevStage         = null;
        this.stageSequenceRep  = [];

        // Yoga
        this.angleHistory    = [];
        this.MAX_ANGLE_HIST  = 20;
        this.holdStartTime   = null;
        this.holdCompleted   = false;
        this.HOLD_DURATION   = 3.0;

        // Error tracking (mirrors coach.py)
        this.lastErrorKey          = null;
        this.consecutiveErrorCount = 0;
        this.ERROR_REPLAY_THRESHOLD= 1;

        this.startTime  = null;
        this.isRunning  = false;

        // Rep quality history (for dots UI)
        this.repQualityHistory = [];
    }

    start() { this.startTime = performance.now(); this.isRunning = true; }
    stop()  { this.isRunning = false; }

    getElapsedTime()   { return this.startTime ? (performance.now()-this.startTime)/1000 : 0; }
    getFormattedTime() {
        const s = Math.floor(this.getElapsedTime());
        return `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
    }

    /**
     * Process a single frame from MediaPipe
     * @param {Array} landmarks  - [{x,y,z,visibility}, ...] (normalized 0-1)
     * @param {number} w         - canvas width
     * @param {number} h         - canvas height
     * @returns {Object}
     */
    processFrame(landmarks, w, h) {
        if (!this.isRunning || !landmarks) return null;

        // Real frame-rate tracking — avoids inflated speed when processing < 30fps
        const now = performance.now();
        let effectiveFPS = this.FPS;
        if (this._lastFrameTime !== null) {
            const dt = (now - this._lastFrameTime) / 1000;
            if (dt > 0) effectiveFPS = Math.min(1 / dt, 60);
        }
        this._lastFrameTime = now;

        const primaryJoint = this.config.primary_angle;
        const jointDef     = JOINTS[primaryJoint];
        if (!jointDef) return null;

        const [ia, ib, ic] = jointDef;
        let a = [landmarks[ia].x*w, landmarks[ia].y*h];
        let b = [landmarks[ib].x*w, landmarks[ib].y*h];
        let c = [landmarks[ic].x*w, landmarks[ic].y*h];
        const localAngle = calculateAngle(a, b, c);
        let angle = localAngle;

        // Track which arm/joint is more engaged — starts as primary, may switch to opposite
        let activeJointIdx = jointDef[1];

        // Opposite-joint tracking — only use if the joint is well-detected
        // (a poorly-detected opposite joint can lock the rep state machine)
        const oppJoint = OPPOSITE_JOINT[primaryJoint];
        if (oppJoint && JOINTS[oppJoint]) {
            const [oa,ob,oc] = JOINTS[oppJoint];
            const oppVis = Math.min(
                landmarks[oa].visibility ?? 1,
                landmarks[ob].visibility ?? 1,
                landmarks[oc].visibility ?? 1
            );
            if (oppVis > 0.35) {
                const oppAngle = calculateAngle(
                    [landmarks[oa].x*w, landmarks[oa].y*h],
                    [landmarks[ob].x*w, landmarks[ob].y*h],
                    [landmarks[oc].x*w, landmarks[oc].y*h]
                );
                if (this.config.direction === 'decrease_then_increase' || this.config.type === 'yoga') {
                    if (oppAngle < localAngle) activeJointIdx = JOINTS[oppJoint][1];
                    angle = Math.min(localAngle, oppAngle);
                } else {
                    if (oppAngle > localAngle) activeJointIdx = JOINTS[oppJoint][1];
                    angle = Math.max(localAngle, oppAngle);
                }
            }
        }

        // Speed (deg/s) — uses actual frame delta, not fixed 30fps
        let speed = 0;
        if (this.prevAngle !== null) {
            speed = Math.abs(angle - this.prevAngle) * effectiveFPS;
            this.speedBuffer.push(speed);
            if (this.speedBuffer.length > 10) this.speedBuffer.shift();
        }
        this.prevAngle = angle;

        // Per-rep min/max tracking
        if (this.repMinAngle === null) { this.repMinAngle = angle; this.repMaxAngle = angle; }
        this.repMinAngle = Math.min(this.repMinAngle, angle);
        this.repMaxAngle = Math.max(this.repMaxAngle, angle);
        if (speed > 0) this.repSpeedSamples.push(speed);

        // FSM update
        let stage;
        if (this.config.type === 'boxing') {
            const avgSpeed5 = this.speedBuffer.length >= 5
                ? this.speedBuffer.slice(-5).reduce((s,v)=>s+v,0)/5 : 0;
            stage = this.fsm.update(avgSpeed5);
        } else if (this.config.type === 'yoga') {
            this.angleHistory.push(angle);
            if (this.angleHistory.length > this.MAX_ANGLE_HIST) this.angleHistory.shift();
            const mean = this.angleHistory.reduce((s,v)=>s+v,0)/this.angleHistory.length;
            const std  = Math.sqrt(this.angleHistory.reduce((s,v)=>s+(v-mean)**2,0)/this.angleHistory.length);
            stage = this.fsm.update(std);
        } else {
            stage = this.fsm.update(angle);
        }

        // Rep counting
        let repCompleted  = false;
        let repEvaluation = null;

        if (this.config.type === 'yoga') {
            repCompleted = this._checkRepYoga(stage);
        } else if (this.config.type === 'boxing') {
            repCompleted = this._checkRepMultistage(stage);
        } else {
            repCompleted = this._checkRepSingle(angle);
        }

        if (repCompleted) {
            this.repCount++;
            repEvaluation = this.evaluateRep();
            this.repQualityHistory.push(repEvaluation.errorKey ? 'bad' : 'good');
            // Reset per-rep tracking
            this.repMinAngle     = null;
            this.repMaxAngle     = null;
            this.repSpeedSamples = [];
        }

        const quality = this._calcQuality(angle, speed);

        return {
            stage,
            angle:         Math.round(angle * 10) / 10,
            speed:         Math.round(speed),
            repCompleted,
            repCount:      this.repCount,
            repEvaluation,
            quality,
            elapsed:       this.getFormattedTime(),
            activeJointIdx,
        };
    }

    // Mirrors coach.py single-exercise rep counting (the run() loop logic)
    _checkRepSingle(angle) {
        const stats = this.model?.joints?.[this.config.primary_angle];
        if (!stats) return false;

        const rom = Math.max(stats.max - stats.min, 20);
        const isDecrease = (this.config.direction === 'decrease_then_increase');

        const now = performance.now() / 1000;

        if (isDecrease) {
            // Lowered thresholds (20%/10%) so normal-pace reps are detected
            const activeThreshold = stats.max - (rom * 0.20);
            const restThreshold   = stats.max - (rom * 0.10);
            if (angle < activeThreshold && this.repState === 'rest') {
                this.repState = 'active';
            } else if (angle > restThreshold && this.repState === 'active') {
                if (now - this.lastRepTime > this.MIN_REP_GAP) {
                    this.lastRepTime = now;
                    this.repState = 'rest';
                    return true;
                }
            }
        } else {
            const activeThreshold = stats.min + (rom * 0.20);
            const restThreshold   = stats.min + (rom * 0.10);
            if (angle > activeThreshold && this.repState === 'rest') {
                this.repState = 'active';
            } else if (angle < restThreshold && this.repState === 'active') {
                if (now - this.lastRepTime > this.MIN_REP_GAP) {
                    this.lastRepTime = now;
                    this.repState = 'rest';
                    return true;
                }
            }
        }
        return false;
    }

    _checkRepMultistage(stage) {
        const expected = { boxing: ['guard','punch','recover'], yoga: ['entering','hold','exiting'] };
        const seq = expected[this.config.type];
        if (!seq) return false;
        if (stage !== this.prevStage) {
            this.stageSequenceRep.push(stage);
            this.prevStage = stage;
            if (this.stageSequenceRep.length >= seq.length) {
                const recent = this.stageSequenceRep.slice(-seq.length);
                if (JSON.stringify(recent) === JSON.stringify(seq)) {
                    this.stageSequenceRep = [];
                    return true;
                }
            }
        }
        return false;
    }

    _checkRepYoga(stage) {
        const HOLD_DURATION = 3.0;
        if (stage === 'hold') {
            if (this.holdStartTime === null) {
                this.holdStartTime  = performance.now() / 1000;
                this.holdCompleted  = false;
            }
            const holdTime = (performance.now() / 1000) - this.holdStartTime;
            if (holdTime >= HOLD_DURATION && !this.holdCompleted) {
                this.holdCompleted = true;
                return true;
            }
        } else {
            if (this.holdStartTime !== null && !this.holdCompleted) {
                // released too soon — reset
            }
            this.holdStartTime = null;
            this.holdCompleted = false;
        }
        return false;
    }

    _calcQuality(angle, speed) {
        const stats = this.model?.joints?.[this.config.primary_angle];
        if (!stats) return 0;
        const romRange = stats.max - stats.min;
        if (romRange < 1) return 50;

        let romScore   = 100;
        if (angle < stats.min - 15) romScore -= 30;
        if (angle > stats.max + 15) romScore -= 30;

        let speedScore = 100;
        if (stats.speed_avg > 0) {
            const diff = Math.abs(speed - stats.speed_avg);
            if (diff > stats.speed_std * 2) speedScore -= 40;
            else if (diff > stats.speed_std)  speedScore -= 15;
        }

        const q = Math.max(0, Math.min(100, Math.round(romScore * 0.7 + speedScore * 0.3)));
        this.qualityScores.push(q);
        if (this.qualityScores.length > 30) this.qualityScores.shift();
        this.currentQuality = Math.round(this.qualityScores.reduce((a,b)=>a+b,0)/this.qualityScores.length);
        return this.currentQuality;
    }

    getAverageQuality() {
        if (!this.qualityScores.length) return 0;
        return Math.round(this.qualityScores.reduce((a,b)=>a+b,0)/this.qualityScores.length);
    }

    getSummary() {
        return {
            exercise:   this.config.name,
            exerciseId: this.config.id,
            type:       this.config.type,
            reps:       this.repCount,
            duration:   Math.round(this.getElapsedTime()),
            quality:    this.getAverageQuality(),
            icon:       this.config.icon || '🏋️',
        };
    }

    // ═══════════════════════════════════════════════════════════════
    //  evaluateRep — EXACT mirror of coach.py LiveCoach.evaluate_rep()
    //  with full exercise-specific feedback tables from coach.py
    // ═══════════════════════════════════════════════════════════════
    evaluateRep() {
        const stats = this.model?.joints?.[this.config.primary_angle];
        if (!stats) return { message: 'Good rep!', errorKey: null };

        const romTotal  = stats.max - stats.min;
        const actualRom = (this.repMaxAngle !== null && this.repMinAngle !== null)
            ? (this.repMaxAngle - this.repMinAngle) : 0;
        const romRatio  = romTotal > 0 ? actualRom / romTotal : 0;
        const avgSpeed  = this.repSpeedSamples.length > 0
            ? this.repSpeedSamples.reduce((a,b)=>a+b,0)/this.repSpeedSamples.length : 0;

        // ─── Full feedback tables (copied verbatim from coach.py) ─
        const FEEDBACK = {
            squat: {
                rom_very_shallow: "Barely bending — drop your hips well below knee level",
                rom_shallow:      "Go lower! Your thighs should be parallel to the ground",
                rom_moderate:     "Almost there — push your hips back a bit more",
                rom_good:         "Good squat depth, try to go just a touch lower",
                rom_excellent:    "Perfect depth — great squat!",
                speed_too_slow:   "Too slow — push through the squat with more power",
                speed_slow:       "Speed up slightly, drive through your heels",
                speed_fast:       "Slow down — control the descent, don't drop too fast",
                speed_too_fast:   "Way too fast! Slow controlled squat, 2 seconds down",
                speed_good:       "Nice controlled tempo",
                bottom_too_high:  "Your bottom position is too high — break parallel",
                top_not_extended: "Stand up fully at the top, lock your hips out",
                alignment_issue:  "Keep your chest up and back straight",
                praise_full:      "Textbook squat — great depth and control!",
            },
            pushup: {
                rom_very_shallow: "Barely moving — lower your chest all the way to the floor",
                rom_shallow:      "Go deeper! Your chest should nearly touch the ground",
                rom_moderate:     "Lower your body more, elbows to 90 degrees",
                rom_good:         "Good depth, just a little lower for full range",
                rom_excellent:    "Full range pushup — excellent!",
                speed_too_slow:   "Too slow — push up explosively off the ground",
                speed_slow:       "Pick up the pace, drive through your palms",
                speed_fast:       "Slow the descent down — control the lowering phase",
                speed_too_fast:   "Way too fast! 2 seconds down, 1 second up",
                speed_good:       "Great pushup tempo",
                bottom_too_high:  "Your chest is too high — get closer to the floor",
                top_not_extended: "Lock your elbows at the top, full arm extension",
                alignment_issue:  "Keep your core tight, don't let your hips sag",
                praise_full:      "Clean pushup — solid form and depth!",
            },
            bicep_curl: {
                rom_very_shallow: "Barely curling — bring the weight all the way up to your shoulder",
                rom_shallow:      "Curl higher! Bring your forearm up to your bicep",
                rom_moderate:     "Almost full curl — squeeze at the top more",
                rom_good:         "Good curl range, try to peak the contraction higher",
                rom_excellent:    "Full range curl — great bicep engagement!",
                speed_too_slow:   "Too slow — add some energy to the curl",
                speed_slow:       "Curl with a bit more momentum on the way up",
                speed_fast:       "Slow the lowering phase — don't let the weight drop",
                speed_too_fast:   "Way too fast! You're swinging, not curling",
                speed_good:       "Perfect curl tempo — controlled and smooth",
                bottom_too_high:  "Extend your arm fully at the bottom, don't cheat the range",
                top_not_extended: "Squeeze harder at the top, bring it closer to your shoulder",
                alignment_issue:  "Keep your elbows pinned to your sides, don't swing",
                praise_full:      "Perfect curl — full range, no swinging, great squeeze!",
            },
            boxing: {
                rom_very_shallow: "Barely punching — extend your arm fully on each punch",
                rom_shallow:      "Extend your punch further, reach full arm extension",
                rom_moderate:     "Almost full extension, snap your punches more",
                rom_good:         "Good punch reach, try to snap it a bit more",
                rom_excellent:    "Full extension — sharp punches!",
                speed_too_slow:   "Too slow — snap your punches, be explosive",
                speed_slow:       "Faster jabs, speed is power in boxing",
                speed_fast:       "Good speed but don't sacrifice form for pace",
                speed_too_fast:   "Pulling back too fast — return to guard position cleanly",
                speed_good:       "Great punch speed and timing",
                bottom_too_high:  "Extend your punch more — snap your arm out fully",
                top_not_extended: "Return to guard position with hands up by your chin",
                alignment_issue:  "Keep your guard up! Hands protect your face between punches",
                praise_full:      "Sharp combos — great speed, full extension, strong guard!",
            },
            yoga: {
                rom_very_shallow: "Ease deeper into the pose — you're barely in position",
                rom_shallow:      "Go deeper into the stretch, breathe into it",
                rom_moderate:     "Good start, push the pose a little further",
                rom_good:         "Nearly full expression — reach just a bit more",
                rom_excellent:    "Beautiful pose depth — fully expressed!",
                speed_too_slow:   "Flow with more intention, don't stall in transition",
                speed_slow:       "Transition a bit more smoothly between poses",
                speed_fast:       "Slow down — yoga is about control, not speed",
                speed_too_fast:   "Way too fast — breathe deeply, move with your breath",
                speed_good:       "Smooth and controlled — nice flow",
                bottom_too_high:  "Sink deeper into the pose, lengthen through your spine",
                top_not_extended: "Reach fully through the pose, extend your limbs",
                alignment_issue:  "Check your alignment — stack your joints, engage your core",
                praise_full:      "Namaste — beautiful form, stability, and breath control!",
            },
        };

        const DEFAULT_FEEDBACK = {
            rom_very_shallow: "Very shallow movement — use your full range of motion",
            rom_shallow:      "Increase your range of motion significantly",
            rom_moderate:     "Go a bit deeper for full range",
            rom_good:         "Good depth, push just a little more",
            rom_excellent:    "Great range of motion!",
            speed_too_slow:   "Move with more energy",
            speed_slow:       "Speed up a bit",
            speed_fast:       "Slow down, focus on control",
            speed_too_fast:   "Way too fast! Slow and controlled",
            speed_good:       "Good controlled pace",
            bottom_too_high:  "Go lower at the bottom",
            top_not_extended: "Fully extend at the top",
            alignment_issue:  "Watch your body alignment",
            praise_full:      "Great rep — solid form!",
        };

        const fb = FEEDBACK[this.config.id] || DEFAULT_FEEDBACK;

        const issues    = [];
        const positives = [];

        // ─── ROM quality (mirrors coach.py exactly) ───────────────
        if      (romRatio < 0.4)  issues.push({ key:'rom_very_shallow', msg:fb.rom_very_shallow });
        else if (romRatio < 0.6)  issues.push({ key:'rom_shallow',      msg:fb.rom_shallow      });
        else if (romRatio < 0.75) issues.push({ key:'rom_moderate',     msg:fb.rom_moderate     });
        else if (romRatio < 0.85) issues.push({ key:'rom_good',         msg:fb.rom_good         });
        else                      positives.push({ key:null,            msg:fb.rom_excellent     });

        // ─── Speed quality (mirrors coach.py exactly) ─────────────
        const modelSpeed    = stats.speed_avg || 50;
        const modelSpeedStd = stats.speed_std || 30;
        if (avgSpeed < 5) {
            issues.push({ key:'speed_too_slow', msg:fb.speed_too_slow });
        } else if (avgSpeed < modelSpeed * 0.3) {
            issues.push({ key:'speed_slow',     msg:fb.speed_slow });
        } else if (avgSpeed > modelSpeed + modelSpeedStd * 2) {
            issues.push({ key:'speed_too_fast', msg:fb.speed_too_fast });
        } else if (avgSpeed > modelSpeed + modelSpeedStd) {
            issues.push({ key:'speed_fast',     msg:fb.speed_fast });
        } else {
            positives.push({ key:null,          msg:fb.speed_good });
        }

        // ─── Form alignment (mirrors coach.py exactly) ────────────
        if (this.repMinAngle !== null && this.repMaxAngle !== null) {
            if (this.repMinAngle > stats.min + 25)  issues.push({ key:'bottom_too_high',  msg:fb.bottom_too_high });
            if (this.repMaxAngle < stats.max - 25)  issues.push({ key:'top_not_extended', msg:fb.top_not_extended });
        }

        // ─── Alignment check when ROM is very bad ─────────────────
        if (romRatio < 0.5) issues.push({ key:'alignment_issue', msg:fb.alignment_issue });

        // ─── Final decision (mirrors coach.py exactly) ────────────
        if (issues.length > 0) {
            const combined = issues.map(i=>i.msg).join('. ');
            return { message: combined, errorKey: issues[0].key };
        } else if (positives.length >= 2) {
            return { message: fb.praise_full, errorKey: null };
        } else if (positives.length === 1) {
            return { message: positives[0].msg, errorKey: null };
        }
        return { message: fb.praise_full, errorKey: null };
    }
}

// ─── Exports ─────────────────────────────────────────────────────────────────
window.ExerciseSession  = ExerciseSession;
window.JOINTS           = JOINTS;
window.OPPOSITE_JOINT   = OPPOSITE_JOINT;
window.calculateAngle   = calculateAngle;
