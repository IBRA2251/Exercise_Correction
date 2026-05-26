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

        // Portrait dimensions on mobile so the full body fits without heavy crop
        const isMobile = window.innerWidth < 768 || /Mobi|Android/i.test(navigator.userAgent);
        const camW = isMobile ? 480 : 1280;
        const camH = isMobile ? 640 : 720;

        this.camera = new Camera(videoElement, {
            onFrame: async () => {
                if (this.pose) await this.pose.send({ image: videoElement });
            },
            width:  camW,
            height: camH,
        });

        await this.camera.start();
        this.isReady = true;

        // Hide whichever loading overlay is present — coach uses 'video-loading',
        // trainer uses 'trainer-video-loading'
        ['video-loading', 'trainer-video-loading'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
    }

    _handleResults(results) {
        if (!this.canvasEl || !this.ctx) return;

        // Size canvas to its CSS display container so the overlay matches layout
        const cW = this.canvasEl.offsetWidth  || 1280;
        const cH = this.canvasEl.offsetHeight || 720;
        this.canvasEl.width  = cW;
        this.canvasEl.height = cH;

        // Compute object-fit:cover mapping from video space → canvas space
        const vidW = this.videoEl.videoWidth  || 1280;
        const vidH = this.videoEl.videoHeight || 720;
        const scale   = Math.max(cW / vidW, cH / vidH);
        const renderW = vidW * scale;
        const renderH = vidH * scale;
        const offsetX = (cW - renderW) / 2;   // negative = video cropped on sides
        const offsetY = (cH - renderH) / 2;   // negative = video cropped on top/bottom

        const ctx = this.ctx;
        ctx.clearRect(0, 0, cW, cH);

        // FPS
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsTime >= 1000) {
            this.currentFps  = this.frameCount;
            this.frameCount  = 0;
            this.lastFpsTime = now;
            // Support both the coach page (#fps-display) and the trainer page (#trainer-fps-display)
            const el = document.getElementById('fps-display') || document.getElementById('trainer-fps-display');
            if (el) el.textContent = `${this.currentFps} FPS`;
        }

        if (results.poseLandmarks) {
            this._drawSkeleton(ctx, results.poseLandmarks, renderW, renderH, offsetX, offsetY);
        }

        if (this.onResults) {
            // Pass video native size for accurate angle math in processFrame
            this.onResults(results.poseLandmarks || null, vidW, vidH);
        }
    }

    _drawSkeleton(ctx, landmarks, renderW, renderH, offsetX, offsetY) {
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
                    ctx.moveTo(a.x*renderW + offsetX, a.y*renderH + offsetY);
                    ctx.lineTo(b.x*renderW + offsetX, b.y*renderH + offsetY);
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
                const px = lm.x*renderW + offsetX;
                const py = lm.y*renderH + offsetY;
                // Outer glow
                ctx.fillStyle = isPrimary
                    ? 'rgba(245,158,11,0.35)'
                    : 'rgba(6,182,212,0.25)';
                ctx.beginPath();
                ctx.arc(px, py, isPrimary ? 12 : 8, 0, Math.PI*2);
                ctx.fill();
                // Inner point
                ctx.fillStyle = isPrimary ? '#f59e0b' : '#06b6d4';
                ctx.beginPath();
                ctx.arc(px, py, isPrimary ? 6 : 4, 0, Math.PI*2);
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
                ctx.fillText(`${Math.round(this.currentAngle)}°`,
                    lm.x*renderW + offsetX + 14, lm.y*renderH + offsetY - 10);
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
