/**
 * Pose Engine — MediaPipe Pose integration with enhanced canvas overlay
 * Handles webcam access, pose detection, and neon skeleton rendering.
 * Adds: angle arc display, primary-joint highlight, FPS counter.
 */

const POSE_CONNECTIONS_DRAW = [
    [11,13],[13,15],  // Left arm
    [12,14],[14,16],  // Right arm
    [11,12],          // Shoulders
    [23,25],[25,27],  // Left leg
    [24,26],[26,28],  // Right leg
    [23,24],          // Hips
    [11,23],[12,24],  // Torso sides
];

class PoseEngine {
    constructor() {
        this.pose      = null;
        this.camera    = null;
        this.videoEl   = null;
        this.canvasEl  = null;
        this.ctx       = null;
        this.isReady   = false;
        this.onResults = null;

        this.frameCount  = 0;
        this.lastFpsTime = performance.now();
        this.currentFps  = 0;

        // Set from outside to highlight a joint and show angle
        this.primaryJointIdx = null;  // landmark index of the mid-joint
        this.currentAngle    = null;
    }

    async init(videoElement, canvasElement) {
        this.videoEl  = videoElement;
        this.canvasEl = canvasElement;
        this.ctx      = canvasElement.getContext('2d');

        this.pose = new Pose({
            locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`
        });

        this.pose.setOptions({
            modelComplexity:        1,
            smoothLandmarks:        true,
            minDetectionConfidence: 0.6,
            minTrackingConfidence:  0.6,
        });

        this.pose.onResults((r) => this._handleResults(r));

        this.camera = new Camera(videoElement, {
            onFrame: async () => {
                if (this.pose) await this.pose.send({ image: videoElement });
            },
            width:  1280,
            height: 720,
        });

        await this.camera.start();
        this.isReady = true;

        const loading = document.getElementById('video-loading');
        if (loading) loading.style.display = 'none';
    }

    _handleResults(results) {
        if (!this.canvasEl || !this.ctx) return;

        this.canvasEl.width  = this.videoEl.videoWidth  || 1280;
        this.canvasEl.height = this.videoEl.videoHeight || 720;

        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);

        // FPS
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsTime >= 1000) {
            this.currentFps  = this.frameCount;
            this.frameCount  = 0;
            this.lastFpsTime = now;
            const el = document.getElementById('fps-display');
            if (el) el.textContent = `${this.currentFps} FPS`;
        }

        if (results.poseLandmarks) {
            this._drawSkeleton(ctx, results.poseLandmarks,
                this.canvasEl.width, this.canvasEl.height);
        }

        if (this.onResults) {
            this.onResults(
                results.poseLandmarks || null,
                this.canvasEl.width,
                this.canvasEl.height
            );
        }
    }

    _drawSkeleton(ctx, landmarks, w, h) {
        ctx.save();

        // ─── Connections ──────────────────────────────────────────
        ctx.shadowColor = 'rgba(99,102,241,0.5)';
        ctx.shadowBlur  = 14;
        ctx.strokeStyle = 'rgba(99,102,241,0.85)';
        ctx.lineWidth   = 3;
        ctx.lineCap     = 'round';

        for (const [i, j] of POSE_CONNECTIONS_DRAW) {
            if (i < landmarks.length && j < landmarks.length) {
                const a = landmarks[i], b = landmarks[j];
                if ((a.visibility ?? 1) > 0.4 && (b.visibility ?? 1) > 0.4) {
                    ctx.beginPath();
                    ctx.moveTo(a.x*w, a.y*h);
                    ctx.lineTo(b.x*w, b.y*h);
                    ctx.stroke();
                }
            }
        }

        // ─── Joints ───────────────────────────────────────────────
        ctx.shadowColor = 'rgba(6,182,212,0.7)';
        ctx.shadowBlur  = 8;

        for (let i = 11; i < landmarks.length && i <= 28; i++) {
            const lm = landmarks[i];
            if ((lm.visibility ?? 1) > 0.4) {
                const isPrimary = (i === this.primaryJointIdx);
                // Outer glow
                ctx.fillStyle = isPrimary
                    ? 'rgba(245,158,11,0.35)'
                    : 'rgba(6,182,212,0.25)';
                ctx.beginPath();
                ctx.arc(lm.x*w, lm.y*h, isPrimary ? 12 : 8, 0, Math.PI*2);
                ctx.fill();
                // Inner point
                ctx.fillStyle = isPrimary ? '#f59e0b' : '#06b6d4';
                ctx.beginPath();
                ctx.arc(lm.x*w, lm.y*h, isPrimary ? 6 : 4, 0, Math.PI*2);
                ctx.fill();
            }
        }

        // ─── Angle label on primary joint ─────────────────────────
        if (this.primaryJointIdx !== null && this.currentAngle !== null) {
            const lm = landmarks[this.primaryJointIdx];
            if (lm && (lm.visibility ?? 1) > 0.4) {
                ctx.font      = 'bold 14px "JetBrains Mono", monospace';
                ctx.fillStyle = '#f59e0b';
                ctx.shadowColor = 'rgba(245,158,11,0.6)';
                ctx.shadowBlur  = 10;
                ctx.fillText(`${Math.round(this.currentAngle)}°`, lm.x*w + 14, lm.y*h - 10);
            }
        }

        ctx.restore();
    }

    stop() {
        if (this.camera) { this.camera.stop(); this.camera = null; }
        if (this.pose)   { this.pose.close();  this.pose   = null; }
        this.isReady = false;
    }
}

window.PoseEngine = PoseEngine;
