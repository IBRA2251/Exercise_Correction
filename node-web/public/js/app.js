/**
 * FormAI — Main Application (Node.js Edition)
 * SPA router, coaching logic, avatar demo, countdown, UI updates.
 * All coaching logic is identical to coach.py — only the UI layer changes.
 */
(function () {
'use strict';

// ═══════════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════════
const S = {
    page:        'home',
    exercises:   [],
    exercise:    null,
    model:       null,
    motionData:  null,
    poseEngine:  null,
    session:     null,
    feedback:    new FeedbackEngine(),
    timer:       null,
    isReplaying: false,
    errorTimeout:null,
    goodStreak:  0,
    avatarRefRAF:null,
    avatarRefIdx:0,
};

// ═══════════════════════════════════════════════════════
//  PROGRESS BAR
// ═══════════════════════════════════════════════════════
const Progress = {
    bar: null,
    start() {
        if (!this.bar) this.bar = document.getElementById('top-progress-bar');
        if (this.bar) { this.bar.style.width = '30%'; this.bar.style.opacity = '1'; }
    },
    done() {
        if (this.bar) {
            this.bar.style.width = '100%';
            setTimeout(() => { if (this.bar) { this.bar.style.opacity='0'; this.bar.style.width='0'; } }, 400);
        }
    },
};

// ═══════════════════════════════════════════════════════
//  API CLIENT
// ═══════════════════════════════════════════════════════
const API = {
    async get(url)         { const r = await fetch(url);  return r.ok ? r.json() : null; },
    async post(url, data)  {
        const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data) });
        return r.json();
    },
    getExercises()           { return this.get('/api/exercises'); },
    getModel(id)             { return this.get(`/api/models/${id}`); },
    getMotionData(id)        { return this.get(`/api/motions/${id}`); },
    getSessions()            { return this.get('/api/sessions'); },
    saveSession(data)        { return this.post('/api/sessions', data); },
};

// ═══════════════════════════════════════════════════════
//  ROUTER
// ═══════════════════════════════════════════════════════
function navigate(page, params = {}) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const el = document.getElementById(`page-${page}`);
    if (el) { el.classList.add('active'); el.style.animation='none'; el.offsetHeight; el.style.animation=''; }
    // Sync both top nav links and bottom tab bar
    document.querySelectorAll('.nav-link, .bnt-item').forEach(l =>
        l.classList.toggle('active', l.dataset.page === page));
    S.page = page;

    if (page === 'exercises') initExercisePage();
    if (page === 'coach')     initCoachPage(params.exercise);
    if (page === 'dashboard') initDashboardPage();
    if (page === 'admin')     initAdminPage();
}

function handleHash() {
    const hash = location.hash.replace('#', '') || 'home';
    if (hash.startsWith('coach/')) navigate('coach', { exercise: hash.split('/')[1] });
    else navigate(hash);
}

// ═══════════════════════════════════════════════════════
//  HERO — Animated skeleton + stat counters
// ═══════════════════════════════════════════════════════
function initHeroSkeleton() {
    const canvas = document.getElementById('hero-skeleton');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let t = 0;

    const J = {
        head:{x:210,y:72},   neck:{x:210,y:110},
        lSh: {x:165,y:135},  rSh: {x:255,y:135},
        lEl: {x:145,y:200},  rEl: {x:275,y:200},
        lWr: {x:145,y:265},  rWr: {x:275,y:265},
        lHip:{x:180,y:285},  rHip:{x:240,y:285},
        lKn: {x:172,y:365},  rKn: {x:248,y:365},
        lAn: {x:168,y:445},  rAn: {x:252,y:445},
    };

    const lines = [
        ['head','neck'],
        ['neck','lSh'],['neck','rSh'],
        ['lSh','lEl'],['lEl','lWr'],
        ['rSh','rEl'],['rEl','rWr'],
        ['lSh','lHip'],['rSh','rHip'],
        ['lHip','rHip'],
        ['lHip','lKn'],['lKn','lAn'],
        ['rHip','rKn'],['rKn','rAn'],
    ];

    function drawFrame() {
        t += 0.02;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const curl = (Math.sin(t) + 1) * 0.5;
        J.rEl.y = 200 - curl * 28;
        J.rWr.x = 275 + curl * 32;
        J.rWr.y = 265 - curl * 120;

        const sway = Math.sin(t * 0.7) * 3;
        const bth  = Math.sin(t * 1.1) * 2;

        // Glow BG
        const g = ctx.createRadialGradient(210, 270, 30, 210, 270, 220);
        g.addColorStop(0, 'rgba(99,102,241,0.09)');
        g.addColorStop(1, 'rgba(99,102,241,0)');
        ctx.fillStyle = g; ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.save();
        ctx.shadowColor = 'rgba(99,102,241,0.55)'; ctx.shadowBlur = 14;
        ctx.strokeStyle = 'rgba(99,102,241,0.75)';  ctx.lineWidth = 3; ctx.lineCap = 'round';
        lines.forEach(([a,b]) => {
            ctx.beginPath();
            ctx.moveTo(J[a].x+sway, J[a].y+bth);
            ctx.lineTo(J[b].x+sway, J[b].y+bth);
            ctx.stroke();
        });

        ctx.shadowColor = 'rgba(6,182,212,0.7)'; ctx.shadowBlur = 10;
        Object.values(J).forEach(j => {
            ctx.fillStyle = 'rgba(6,182,212,0.25)';
            ctx.beginPath(); ctx.arc(j.x+sway, j.y+bth, 8, 0, Math.PI*2); ctx.fill();
            ctx.fillStyle = '#06b6d4';
            ctx.beginPath(); ctx.arc(j.x+sway, j.y+bth, 4, 0, Math.PI*2); ctx.fill();
        });

        // Angle label
        const angleDeg = Math.round(40 + curl * 100);
        ctx.font = 'bold 15px "JetBrains Mono",monospace';
        ctx.fillStyle = '#f59e0b'; ctx.shadowColor = 'rgba(245,158,11,0.5)'; ctx.shadowBlur = 8;
        ctx.fillText(`${angleDeg}°`, J.rEl.x+sway+16, J.rEl.y+bth-10);

        ctx.restore();
        requestAnimationFrame(drawFrame);
    }
    drawFrame();
}

function animateStatCounters() {
    document.querySelectorAll('.hero-stat-value[data-count]').forEach(el => {
        const target  = parseInt(el.dataset.count, 10);
        const dur     = 1800;
        const start   = performance.now();
        const tick    = (now) => {
            const p = Math.min((now - start) / dur, 1);
            el.textContent = Math.round(p * target) + (el.dataset.count === '100' ? '' : '+');
            if (p < 1) requestAnimationFrame(tick);
            else el.textContent = target + (el.dataset.count === '33' || el.dataset.count === '30' ? '' : '+');
        };
        setTimeout(() => requestAnimationFrame(tick), 400);
    });
}

// ═══════════════════════════════════════════════════════
//  PARTICLE BACKGROUND
// ═══════════════════════════════════════════════════════
function initParticles() {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const N   = 55;
    let particles = [];

    function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
    resize(); window.addEventListener('resize', resize);

    class P {
        constructor() { this.reset(); }
        reset() {
            this.x  = Math.random() * canvas.width;
            this.y  = Math.random() * canvas.height;
            this.sz = Math.random() * 2 + 0.4;
            this.vx = (Math.random() - 0.5) * 0.35;
            this.vy = (Math.random() - 0.5) * 0.35;
            this.op = Math.random() * 0.35 + 0.05;
        }
        update() {
            this.x += this.vx; this.y += this.vy;
            if (this.x < 0 || this.x > canvas.width)  this.vx *= -1;
            if (this.y < 0 || this.y > canvas.height)  this.vy *= -1;
        }
        draw() {
            ctx.beginPath(); ctx.arc(this.x, this.y, this.sz, 0, Math.PI*2);
            ctx.fillStyle = `rgba(99,102,241,${this.op})`; ctx.fill();
        }
    }

    for (let i = 0; i < N; i++) particles.push(new P());

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < particles.length; i++) {
            for (let j = i+1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const d  = Math.sqrt(dx*dx+dy*dy);
                if (d < 140) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(99,102,241,${0.07*(1-d/140)})`;
                    ctx.lineWidth = 0.5; ctx.stroke();
                }
            }
        }
        particles.forEach(p => { p.update(); p.draw(); });
        requestAnimationFrame(animate);
    }
    animate();
}

// ═══════════════════════════════════════════════════════
//  EXERCISE SELECTION PAGE
// ═══════════════════════════════════════════════════════
async function initExercisePage() {
    if (!S.exercises.length) {
        Progress.start();
        S.exercises = await API.getExercises() || [];
        Progress.done();
    }
    renderExerciseCards(S.exercises);
    setupFilters();
}

function renderExerciseCards(list) {
    const grid = document.getElementById('exercise-grid');
    if (!grid) return;
    grid.innerHTML = list.map((ex, i) => `
        <div class="exercise-card" data-id="${ex.id}" data-cat="${ex.category}"
             style="--card-color:${ex.color||'var(--grad-primary)'}; --card-glow:${hex2rgba(ex.color||'#6366f1',0.08)}; animation-delay:${i*80}ms">
            <div class="exercise-card-glow"></div>
            <div class="exercise-card-header">
                <div class="exercise-icon">${ex.icon}</div>
                <div>
                    <div class="exercise-card-title">${ex.name}</div>
                    <div class="exercise-card-badges">
                        <span class="badge badge-category">${ex.category}</span>
                        <span class="badge badge-difficulty">${ex.difficulty}</span>
                    </div>
                </div>
            </div>
            <p class="exercise-card-desc">${ex.description}</p>
            <div class="exercise-muscles">${ex.muscles.map(m=>`<span class="muscle-tag">${m}</span>`).join('')}</div>
            <div class="exercise-card-cta">Start Training →</div>
        </div>
    `).join('');

    grid.querySelectorAll('.exercise-card').forEach((card, i) => {
        card.style.opacity = '0'; card.style.transform = 'translateY(18px)';
        setTimeout(() => {
            card.style.transition = 'opacity .35s ease, transform .35s ease';
            card.style.opacity = '1'; card.style.transform = 'translateY(0)';
        }, i * 70);
        card.addEventListener('click', () => { location.hash = `coach/${card.dataset.id}`; });
    });
}

function setupFilters() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const cat = btn.dataset.category;
            renderExerciseCards(cat === 'all' ? S.exercises : S.exercises.filter(e => e.category === cat));
        });
    });
}

function hex2rgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
    return `rgba(${r},${g},${b},${alpha})`;
}

// ═══════════════════════════════════════════════════════
//  COACHING PAGE — INIT
// ═══════════════════════════════════════════════════════
async function initCoachPage(exerciseId) {
    if (!exerciseId) { location.hash = 'exercises'; return; }
    if (!S.exercises.length) {
        S.exercises = await API.getExercises() || [];
    }
    S.exercise = S.exercises.find(e => e.id === exerciseId);
    if (!S.exercise) { location.hash = 'exercises'; return; }

    Progress.start();
    S.model = await API.getModel(exerciseId);
    Progress.done();

    // Reset UI — desktop + mobile
    setText('coach-exercise-name', S.exercise.name);
    setText('rep-count',         '0');   setText('m-rep-count', '0');
    setText('quality-score',     '--');  setText('m-quality',   '--');
    setText('quality-label',     'Start exercising to see your score');
    setText('metric-stage',      '—');   setText('m-stage',  '—');
    setText('metric-angle',      '—°'); setText('m-angle',  '—°');
    setText('metric-speed',      '—');   setText('m-speed',  '—');
    setText('metric-timer',      '00:00'); setText('m-timer','00:00');
    setText('session-status-badge', 'IDLE');
    setClass('session-status-badge', '');
    const qBar = document.getElementById('quality-bar-fill');
    if (qBar) qBar.style.width = '0%';
    const ringEl = document.getElementById('rep-ring-fill');
    if (ringEl) ringEl.style.strokeDashoffset = '326.73';
    document.getElementById('session-progress-bar').style.width = '0%';
    document.querySelectorAll('.rep-dot').forEach(d => { d.className = 'rep-dot'; });
    hideElement('streak-badge'); hideElement('avatar-ref-panel'); hideElement('btn-replay-demo');
    S.goodStreak = 0;

    // Buttons
    show('btn-start-session'); hide('btn-stop-session');

    // Feedback log
    document.getElementById('feedback-log').innerHTML =
        '<div class="feedback-empty">Feedback appears here during your session</div>';

    // Show loading while camera starts
    show('video-loading');

    // Init / re-init pose engine
    if (S.poseEngine) { S.poseEngine.stop(); S.poseEngine = null; }
    S.poseEngine = new PoseEngine();

    // Set primary joint for angle overlay
    const jDef = JOINTS[S.exercise.primary_angle];
    if (jDef) S.poseEngine.primaryJointIdx = jDef[1]; // mid joint

    try {
        await S.poseEngine.init(
            document.getElementById('webcam-video'),
            document.getElementById('pose-canvas')
        );
    } catch (err) {
        document.getElementById('video-loading').innerHTML = `
            <div style="text-align:center;padding:2rem">
                <div style="font-size:3rem;margin-bottom:1rem">📷</div>
                <h3 style="margin-bottom:.5rem">Camera Access Required</h3>
                <p style="color:var(--text-secondary);margin-bottom:1rem">
                    Allow camera access to use the AI coach.
                    Refresh the page after granting permission.
                </p>
                <a href="#exercises" class="btn btn-glass">← Back to Exercises</a>
            </div>`;
    }

    S.feedback.reset();
    drawQualityRing(0);
}

// ═══════════════════════════════════════════════════════
//  SESSION — START
// ═══════════════════════════════════════════════════════
async function startSession() {
    if (!S.exercise) return;

    S.session    = new ExerciseSession(S.exercise, S.model);
    S.feedback.reset();
    S.feedback.unlockAudio();   // unlock synth from this user-gesture context
    S.isReplaying= false;
    S.goodStreak = 0;
    hideElement('streak-badge');

    // Load motion data for avatar demo
    if (!S.motionData) {
        S.motionData = await API.getMotionData(S.exercise.id);
    }

    // Wire auto-replay
    S.feedback.onReplayTriggered = (msg) => triggerAutoReplay(msg);

    // EMA display-smoothing state (only for UI display, not rep-counting logic)
    let _dispAngle = 0;
    let _dispSpeed = 0;
    const DISP_ALPHA_ANGLE = 0.28;   // responsiveness vs. smoothness
    const DISP_ALPHA_SPEED = 0.18;

    // Wire pose engine
    S.poseEngine.onResults = (landmarks, w, h) => {
        if (!S.session || !S.session.isRunning || S.isReplaying) return;

        const result = S.session.processFrame(landmarks, w, h);
        if (!result) return;

        // Smooth angle + speed for display only — rep counting uses raw result
        if (_dispAngle === 0) _dispAngle = result.angle;  // seed on first frame
        if (_dispSpeed === 0) _dispSpeed = result.speed;
        _dispAngle = _dispAngle * (1 - DISP_ALPHA_ANGLE) + result.angle * DISP_ALPHA_ANGLE;
        _dispSpeed = _dispSpeed * (1 - DISP_ALPHA_SPEED) + result.speed * DISP_ALPHA_SPEED;

        // Update canvas angle label with the smoothed value (less jitter)
        if (S.poseEngine) S.poseEngine.currentAngle = _dispAngle;

        const displayResult = Object.assign({}, result, {
            angle: Math.round(_dispAngle * 10) / 10,
            speed: Math.round(_dispSpeed),
        });
        updateCoachUI(displayResult);

        // Per-frame feedback with cooldown
        const modelStats = S.model?.joints?.[S.exercise.primary_angle];
        const fb = S.feedback.generateFeedback(S.exercise.type, result.stage, result.angle, result.speed, modelStats);
        if (fb) S.feedback.deliver(fb.message);

        // Per-rep evaluation on completion
        if (result.repCompleted && result.repEvaluation) {
            const ev = result.repEvaluation;
            celebrateRep(result.repCount);
            updateRepDots(result.repCount - 1, ev.errorKey ? 'bad' : 'good');
            checkAchievement(result.repCount);

            if (ev.errorKey) {
                showErrorBanner(ev.message);
                S.feedback.deliver(ev.message);
                S.feedback.trackError(ev.errorKey, ev.message);
                S.goodStreak = 0; hideElement('streak-badge');
            } else {
                S.feedback.deliver(ev.message, true);
                S.feedback.resetErrorTracking();
                S.goodStreak++;
                if (S.goodStreak >= 3) showStreakBadge(S.goodStreak);
            }
        }
    };

    S.session.start();

    // Timer
    S.timer = setInterval(() => {
        if (S.session) {
            const t = S.session.getFormattedTime();
            setText('metric-timer', t);
            setText('m-timer', t);
            const prog = Math.min(S.session.repCount / 10, 1) * 100;
            document.getElementById('session-progress-bar').style.width = prog + '%';
        }
    }, 1000);

    // Start avatar reference panel
    startAvatarRef();

    // Status badge
    setText('session-status-badge', 'LIVE');
    setClass('session-status-badge', 'running');
    show('btn-stop-session'); hide('btn-start-session');
    show('btn-replay-demo');
}

// ═══════════════════════════════════════════════════════
//  SESSION — STOP
// ═══════════════════════════════════════════════════════
function stopSession() {
    if (!S.session) return;
    S.session.stop();
    clearInterval(S.timer);
    stopAvatarRef();

    if (S.poseEngine) S.poseEngine.onResults = null;

    setText('session-status-badge', 'DONE');
    setClass('session-status-badge', '');
    hide('btn-stop-session'); show('btn-start-session');
    hide('btn-replay-demo'); hideElement('streak-badge'); hideElement('avatar-ref-panel');

    const summary = S.session.getSummary();
    API.saveSession(summary);
    showSessionModal(summary);
}

// ═══════════════════════════════════════════════════════
//  UI UPDATES
// ═══════════════════════════════════════════════════════
function updateCoachUI(result) {
    const repStr   = String(result.repCount);
    const stageStr = result.stage?.toUpperCase() || '—';
    const angleStr = `${result.angle}°`;
    const speedStr = result.speed ? `${result.speed}°/s` : '—';

    // ── Mobile widgets ──────────────────────────────────
    setText('m-rep-count', repStr);
    setText('m-stage',     stageStr);
    setText('m-angle',     angleStr);
    setText('m-speed',     speedStr);
    if (result.quality > 0) setText('m-quality', `${result.quality}%`);

    // ── Desktop widgets ─────────────────────────────────
    setText('rep-count', repStr);
    const prog = Math.min(result.repCount / 10, 1);
    const ringEl = document.getElementById('rep-ring-fill');
    if (ringEl) ringEl.style.strokeDashoffset = 326.73 * (1 - prog);

    if (result.quality > 0) {
        setText('quality-score', `${result.quality}%`);
        const qBar = document.getElementById('quality-bar-fill');
        if (qBar) qBar.style.width = `${result.quality}%`;
        let lbl = 'Needs improvement';
        if (result.quality >= 90) lbl = '🔥 Excellent form!';
        else if (result.quality >= 75) lbl = '👍 Good form';
        else if (result.quality >= 55) lbl = 'Keep working on it';
        setText('quality-label', lbl);
        drawQualityRing(result.quality);
    }

    setText('metric-stage', stageStr);
    setText('metric-angle', angleStr);
    setText('metric-speed', speedStr);

    const modelStats = S.model?.joints?.[S.exercise.primary_angle];
    if (modelStats) {
        const maxSpeed = (modelStats.speed_avg || 100) + (modelStats.speed_std || 50) * 2;
        const fillPct  = Math.min(result.speed / maxSpeed, 1) * 100;
        const fillEl   = document.getElementById('speed-gauge-fill');
        const needleEl = document.getElementById('speed-gauge-needle');
        if (fillEl)   fillEl.style.width  = `${fillPct}%`;
        if (needleEl) needleEl.style.left = `${fillPct}%`;
    }
}

function updateRepDots(idx, quality) {
    const cls = `rep-dot done-${quality === 'good' ? 'good' : quality === 'warn' ? 'warn' : 'bad'}`;
    // Update both desktop (#rep-dots) and mobile (#m-rep-dots) containers
    document.querySelectorAll(`#rep-dots .rep-dot[data-idx="${idx % 10}"],
        #m-rep-dots .rep-dot[data-idx="${idx % 10}"]`).forEach(d => { d.className = cls; });
}

function showStreakBadge(count) {
    setText('streak-count', count);
    showElement('streak-badge');
}

function drawQualityRing(pct) {
    const canvas = document.getElementById('quality-ring-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const r   = 26, cx = 32, cy = 32;
    ctx.clearRect(0, 0, 64, 64);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth   = 5;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2); ctx.stroke();

    const end = (-Math.PI/2) + (Math.PI*2 * pct/100);
    const g   = ctx.createLinearGradient(0, 0, 64, 0);
    g.addColorStop(0, '#6366f1'); g.addColorStop(1, '#06b6d4');
    ctx.strokeStyle = g; ctx.lineWidth = 5; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.arc(cx, cy, r, -Math.PI/2, end); ctx.stroke();
}

// ═══════════════════════════════════════════════════════
//  ERROR BANNER
// ═══════════════════════════════════════════════════════
function showErrorBanner(message) {
    const banner = document.getElementById('error-banner');
    const text   = document.getElementById('error-banner-text');
    if (!banner || !text) return;
    text.textContent = message;
    banner.style.display = 'flex';
    banner.style.animation = 'none'; banner.offsetHeight; banner.style.animation = '';
    clearTimeout(S.errorTimeout);
    S.errorTimeout = setTimeout(() => { banner.style.display = 'none'; }, 2500);
}

function hideErrorBanner() {
    const banner = document.getElementById('error-banner');
    if (banner) banner.style.display = 'none';
    clearTimeout(S.errorTimeout);
}

// ═══════════════════════════════════════════════════════
//  ACHIEVEMENT TOASTS
// ═══════════════════════════════════════════════════════
const ACHIEVEMENTS = {
    1:  { icon:'🎯', text:'First Rep!' },
    5:  { icon:'🔥', text:'5 Reps!' },
    10: { icon:'💪', text:'10 Reps!' },
    15: { icon:'🏆', text:'15 Reps!' },
    20: { icon:'🌟', text:'20 Reps!' },
    25: { icon:'⚡', text:'25 Reps!' },
};

function checkAchievement(count) {
    const a = ACHIEVEMENTS[count];
    if (!a) return;
    setText('achievement-icon', a.icon);
    setText('achievement-text', a.text);
    const el = document.getElementById('achievement-toast');
    if (!el) return;
    el.style.display = 'flex';
    el.style.animation = 'none'; el.offsetHeight; el.style.animation = '';
    setTimeout(() => { el.style.display = 'none'; }, 2200);
}

// ═══════════════════════════════════════════════════════
//  AUTO-REPLAY (mirrors coach.py auto-replay flow)
// ═══════════════════════════════════════════════════════
async function triggerAutoReplay(errorMsg) {
    if (S.isReplaying) return;
    S.isReplaying = true;
    hideErrorBanner();
    stopAvatarRef();

    // Voice warning
    if (S.feedback.audioEnabled && S.feedback.synth) {
        S.feedback.synth.cancel();
        const utt = new SpeechSynthesisUtterance(
            'You keep making the same mistake. Watch the demonstration again.'
        );
        utt.rate = 1.0;
        S.feedback.synth.speak(utt);
    }

    // Show warning overlay (2.5s — mirrors coach.py warn_start)
    const warnOverlay = document.getElementById('replay-warning-overlay');
    const warnError   = document.getElementById('replay-warning-error');
    if (warnOverlay && warnError) { warnError.textContent = errorMsg; warnOverlay.style.display = 'flex'; }
    await sleep(2500);
    if (warnOverlay) warnOverlay.style.display = 'none';

    // Avatar demo (mirrors coach.py _run_demo)
    if (S.motionData && S.motionData.length) await playAvatarDemo();

    // Countdown 3-2-1-GO (mirrors coach.py _run_countdown)
    await showCountdown();

    S.isReplaying = false;
    S.feedback.resetErrorTracking();
    startAvatarRef();
}

// ═══════════════════════════════════════════════════════
//  3-D MANNEQUIN RENDERER  (Three.js WebGL)
// ═══════════════════════════════════════════════════════
class MannequinRenderer {
    constructor(canvas, w, h) {
        this.w = w; this.h = h;

        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setSize(w, h, false);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.setClearColor(0x080c18, 1);

        this.scene  = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 50);
        this.camera.position.set(0, 0, 3.5);

        // Lighting — key + fill + rim for sculptural depth
        this.scene.add(new THREE.AmbientLight(0x182840, 1.5));
        const key = new THREE.DirectionalLight(0x88aaff, 2.5);
        key.position.set(1.5, 2.5, 3);  this.scene.add(key);
        const fill = new THREE.DirectionalLight(0x2244aa, 0.8);
        fill.position.set(-2, 0.5, 1);  this.scene.add(fill);
        const rim = new THREE.DirectionalLight(0x4477ee, 1.0);
        rim.position.set(0, -2, -1.5);  this.scene.add(rim);

        this._bodyMat = new THREE.MeshPhongMaterial({
            color: 0x1a4a8a, shininess: 90,
            specular: new THREE.Color(0x5090e0),
            emissive: new THREE.Color(0x030a18),
        });
        this._jointMat = new THREE.MeshPhongMaterial({
            color: 0x0d2550, shininess: 130,
            specular: new THREE.Color(0x5588cc),
            emissive: new THREE.Color(0x010408),
        });

        // Segment definitions: [key, fromIdx, toIdx, cylinderRadius]
        this._segDefs = [
            ['lUA',11,13,0.055],['lFA',13,15,0.042],
            ['rUA',12,14,0.055],['rFA',14,16,0.042],
            ['lTh',23,25,0.065],['lSh',25,27,0.050],
            ['rTh',24,26,0.065],['rSh',26,28,0.050],
            ['torL',11,23,0.048],['torR',12,24,0.048],
            ['shld',11,12,0.048],['hips',23,24,0.055],
        ];

        // Unit cylinder (r=1, h=1) shared geometry — scale + rotate per segment
        const cylGeo = new THREE.CylinderGeometry(1, 1, 1, 14, 1);
        this._segMeshes = {};
        for (const [key] of this._segDefs) {
            const m = new THREE.Mesh(cylGeo, this._bodyMat);
            this._segMeshes[key] = m;
            this.scene.add(m);
        }

        // Head sphere
        this._headMesh = new THREE.Mesh(
            new THREE.SphereGeometry(0.1, 16, 12), this._bodyMat);
        this.scene.add(this._headMesh);

        // Joint caps — unit sphere scaled per joint
        const sphGeo = new THREE.SphereGeometry(1, 10, 8);
        this._jointMeshes = {};
        for (const idx of [11,12,13,14,15,16,23,24,25,26,27,28]) {
            const m = new THREE.Mesh(sphGeo, this._jointMat);
            m.visible = false;
            this._jointMeshes[idx] = m;
            this.scene.add(m);
        }

        this._up   = new THREE.Vector3(0, 1, 0);
        this._qTmp = new THREE.Quaternion();
        this._vTmp = new THREE.Vector3();
        this._axX  = new THREE.Vector3(1, 0, 0);
    }

    _lm3D(lm) {
        const asp = this.w / this.h;
        return new THREE.Vector3(
            (lm[0] - 0.5) * 2.1 * asp,
            -(lm[1] - 0.5) * 2.1,
            0
        );
    }

    update(landmarks) {
        if (!landmarks || !landmarks.length) return;
        const pts = landmarks.map(lm => this._lm3D(lm));

        for (const [key, a, b, r] of this._segDefs) {
            if (a >= pts.length || b >= pts.length) continue;
            const p1 = pts[a], p2 = pts[b];
            const dir = this._vTmp.subVectors(p2, p1);
            const len = dir.length();
            if (len < 0.004) continue;

            const mesh = this._segMeshes[key];
            mesh.position.set((p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5, 0);
            mesh.scale.set(r, len, r);

            const norm = dir.clone().normalize();
            if (norm.dot(this._up) < -0.9999) {
                this._qTmp.setFromAxisAngle(this._axX, Math.PI);
            } else {
                this._qTmp.setFromUnitVectors(this._up, norm);
            }
            mesh.quaternion.copy(this._qTmp);
        }

        if (pts.length > 0) this._headMesh.position.copy(pts[0]);

        const jrMap = { 11:0.06, 12:0.06, 23:0.07, 24:0.07 };
        for (const [is, mesh] of Object.entries(this._jointMeshes)) {
            const i = parseInt(is);
            if (i < pts.length) {
                mesh.scale.setScalar(jrMap[i] || 0.048);
                mesh.position.copy(pts[i]);
                mesh.visible = true;
            }
        }
    }

    render() { if (this.renderer) this.renderer.render(this.scene, this.camera); }

    dispose() {
        if (!this.renderer) return;
        this.scene.traverse(o => { if (o.geometry) o.geometry.dispose(); });
        this._bodyMat.dispose();
        this._jointMat.dispose();
        this.renderer.dispose();
        this.renderer = null;
    }
}

// ═══════════════════════════════════════════════════════
//  AVATAR DEMO PLAYBACK
// ═══════════════════════════════════════════════════════
function playAvatarDemo() {
    return new Promise(resolve => {
        const overlay  = document.getElementById('avatar-replay-overlay');
        const canvas   = document.getElementById('avatar-replay-canvas');
        const repsLbl  = document.getElementById('avatar-replay-reps');
        if (!overlay || !canvas) { resolve(); return; }

        overlay.style.display = 'block';
        const rect = overlay.getBoundingClientRect();
        const w = Math.round(rect.width)  || 640;
        const h = Math.round(rect.height) || 480;
        canvas.width = w; canvas.height = h;

        let mr;
        try { mr = new MannequinRenderer(canvas, w, h); }
        catch(e) { resolve(); return; }

        const motion     = S.motionData;
        const exCfg      = S.exercise;
        const modelStats = S.model?.joints?.[exCfg.primary_angle];
        const jDef       = JOINTS[exCfg.primary_angle];

        let frameIdx = 0, demoReps = 0, demoRepState = 'top', raf = null;
        const MAX_REPS = 2;

        function drawFrame() {
            if (demoReps >= MAX_REPS || frameIdx >= motion.length * 2) {
                if (raf) cancelAnimationFrame(raf);
                mr.dispose();
                overlay.style.display = 'none';
                resolve(); return;
            }
            const lm = motion[frameIdx % motion.length];
            mr.update(lm); mr.render();

            if (jDef && modelStats) {
                const [ia,ib,ic] = jDef;
                if (ia < lm.length && ib < lm.length && ic < lm.length) {
                    const angle = calculateAngle([lm[ia][0],lm[ia][1]],[lm[ib][0],lm[ib][1]],[lm[ic][0],lm[ic][1]]);
                    if (angle < modelStats.min + 10 && demoRepState === 'top')         { demoRepState = 'bottom'; }
                    else if (angle > modelStats.max - 10 && demoRepState === 'bottom') { demoRepState = 'top'; demoReps++; }
                }
            }
            if (repsLbl) repsLbl.textContent = `${demoReps}/${MAX_REPS} reps`;

            frameIdx++;
            raf = requestAnimationFrame(drawFrame);
        }
        raf = requestAnimationFrame(drawFrame);
    });
}

// ═══════════════════════════════════════════════════════
//  AVATAR REFERENCE (persistent corner panel during session)
// ═══════════════════════════════════════════════════════
function startAvatarRef() {
    if (!S.motionData || !S.motionData.length) return;
    const panel  = document.getElementById('avatar-ref-panel');
    const canvas = document.getElementById('avatar-ref-canvas');
    if (!panel || !canvas) return;

    panel.style.display = 'block';
    const w = panel.offsetWidth  || 180;
    const h = panel.offsetHeight || 150;
    canvas.width = w; canvas.height = h;

    let mr;
    try { mr = new MannequinRenderer(canvas, w, h); S.avatarRefMR = mr; }
    catch(e) { return; }

    let idx = 0;
    function loop() {
        S.avatarRefRAF = requestAnimationFrame(loop);
        const lm = S.motionData[idx % S.motionData.length];
        mr.update(lm); mr.render();
        idx = (idx + 1) % S.motionData.length;
    }
    requestAnimationFrame(loop);
}

function stopAvatarRef() {
    if (S.avatarRefRAF) { cancelAnimationFrame(S.avatarRefRAF); S.avatarRefRAF = null; }
    if (S.avatarRefMR)  { S.avatarRefMR.dispose(); S.avatarRefMR = null; }
    const panel = document.getElementById('avatar-ref-panel');
    if (panel) panel.style.display = 'none';
}

// ═══════════════════════════════════════════════════════
//  COUNTDOWN 3-2-1-GO (mirrors coach.py _run_countdown)
// ═══════════════════════════════════════════════════════
async function showCountdown() {
    const overlay = document.getElementById('countdown-overlay');
    const numEl   = document.getElementById('countdown-number');
    const subEl   = document.getElementById('countdown-sub');
    const ringFill= document.getElementById('countdown-ring-fill');
    if (!overlay || !numEl) return;

    overlay.style.display = 'flex';

    const steps = [
        { num:'3', sub:'Get Ready...',        ring:326.73 },
        { num:'2', sub:'Position yourself...', ring:218 },
        { num:'1', sub:'Almost there...',      ring:109 },
        { num:'GO!', sub:'Start exercising!',  ring:0 },
    ];

    for (const step of steps) {
        numEl.textContent = step.num;
        if (subEl) subEl.textContent = step.sub;
        if (ringFill) ringFill.style.strokeDashoffset = step.ring;
        numEl.style.animation = 'none'; numEl.offsetHeight; numEl.style.animation = '';
        await sleep(step.num === 'GO!' ? 800 : 1000);
    }

    overlay.style.display = 'none';

    if (S.feedback.audioEnabled && S.feedback.synth) {
        const utt = new SpeechSynthesisUtterance('Go! Start exercising!');
        utt.rate = 1.1;
        S.feedback.synth.speak(utt);
    }
}

// ═══════════════════════════════════════════════════════
//  SESSION MODAL + CONFETTI
// ═══════════════════════════════════════════════════════
function showSessionModal(summary) {
    const modal = document.getElementById('session-modal');
    modal.style.display = 'flex';

    setText('modal-reps', summary.reps);
    const m = Math.floor(summary.duration/60), s = summary.duration%60;
    setText('modal-duration', `${m}:${String(s).padStart(2,'0')}`);
    setText('modal-quality', `${summary.quality}%`);

    const qBar = document.getElementById('modal-quality-bar-fill');
    if (qBar) { qBar.style.width = '0%'; setTimeout(() => { qBar.style.width = `${summary.quality}%`; }, 100); }

    let subtitle = 'Great work — keep it up!';
    if (summary.reps === 0)     subtitle = 'No reps recorded. Try again!';
    else if (summary.quality >= 85) subtitle = 'Outstanding form — you\'re a natural!';
    else if (summary.quality >= 70) subtitle = 'Great session — form is improving!';
    setText('modal-subtitle', subtitle);

    spawnConfetti();
}

function spawnConfetti() {
    const container = document.getElementById('modal-confetti');
    if (!container) return;
    container.innerHTML = '';
    const colors = ['#6366f1','#8b5cf6','#06b6d4','#10b981','#f59e0b','#f43f5e'];
    for (let i = 0; i < 40; i++) {
        const p = document.createElement('div');
        p.className = 'confetti-particle';
        p.style.cssText = `
            left:${Math.random()*100}%;
            top:${-5}px;
            background:${colors[Math.floor(Math.random()*colors.length)]};
            transform:rotate(${Math.random()*360}deg);
            animation-delay:${Math.random()*.8}s;
            animation-duration:${1.5+Math.random()}s;
        `;
        container.appendChild(p);
    }
}

// ═══════════════════════════════════════════════════════
//  DASHBOARD PAGE
// ═══════════════════════════════════════════════════════
async function initDashboardPage() {
    Progress.start();
    const sessions = await API.getSessions() || [];
    Progress.done();

    const total   = sessions.length;
    const reps    = sessions.reduce((s,x)=>s+(x.reps||0), 0);
    const time    = sessions.reduce((s,x)=>s+(x.duration||0), 0);
    const avgQ    = total ? Math.round(sessions.reduce((s,x)=>s+(x.quality||0),0)/total) : 0;

    setText('dash-total-sessions', total);
    setText('dash-total-reps', reps);
    setText('dash-total-time', time >= 60 ? `${Math.floor(time/60)}m` : `${time}s`);
    setText('dash-avg-quality', avgQ > 0 ? `${avgQ}%` : '--');

    renderQualityTrend(sessions.slice(0, 8).reverse());
    renderSessionHistory(sessions);
}

function renderQualityTrend(sessions) {
    const container = document.getElementById('quality-trend-chart');
    if (!container) return;
    if (!sessions.length) {
        container.innerHTML = '<div class="chart-empty">Complete sessions to see your trend</div>';
        return;
    }
    const maxQ   = 100;
    const maxH   = 80;
    container.innerHTML = sessions.map(s => {
        const h   = Math.max(4, Math.round((s.quality||0) / maxQ * maxH));
        const ex  = S.exercises.find(e => e.id === s.exerciseId) || {};
        return `<div class="chart-bar-group">
            <div class="chart-bar" style="height:${h}px" title="${s.exercise||''} · ${s.quality||0}% quality"></div>
            <div class="chart-bar-label">${(ex.icon||'🏋️')}</div>
        </div>`;
    }).join('');
}

function renderSessionHistory(sessions) {
    const container = document.getElementById('session-history');
    if (!container) return;

    if (!sessions.length) {
        container.innerHTML = `<div class="empty-state">
            <div class="empty-icon">📋</div>
            <h3>No sessions yet</h3>
            <p>Complete a workout to see your history here</p>
            <a href="#exercises" class="btn btn-primary">Start Your First Session</a>
        </div>`;
        return;
    }

    const icons = {};
    S.exercises.forEach(e => { icons[e.id] = e.icon; });

    container.innerHTML = sessions.map(s => {
        const icon    = s.icon || icons[s.exerciseId] || '🏋️';
        const mins    = Math.floor((s.duration||0)/60);
        const secs    = (s.duration||0)%60;
        const durStr  = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
        return `<div class="session-card">
            <div class="session-card-left">
                <div class="session-icon">${icon}</div>
                <div class="session-info">
                    <h4>${s.exercise||s.exerciseId||'Exercise'}</h4>
                    <p>${s.timestamp||'Recent'}</p>
                </div>
            </div>
            <div class="session-card-right">
                <div class="session-metric">
                    <span class="session-metric-value">${s.reps||0}</span>
                    <span class="session-metric-label">Reps</span>
                </div>
                <div class="session-metric">
                    <span class="session-metric-value">${durStr}</span>
                    <span class="session-metric-label">Duration</span>
                </div>
                <div class="session-metric">
                    <span class="session-metric-value">${s.quality||0}%</span>
                    <span class="session-metric-label">Quality</span>
                </div>
            </div>
        </div>`;
    }).join('');
}

// ═══════════════════════════════════════════════════════
//  ADMIN TRAINER PAGE
// ═══════════════════════════════════════════════════════

// Extra API call for saving trained baseline
API.saveBaseline = function(exerciseId, model, motion) {
    return this.post(`/api/train/${exerciseId}`, { model, motion });
};

// Check which baselines exist
API.checkBaseline = function(exerciseId) { return this.get(`/api/models/${exerciseId}`); };

const AT = {
    trainer:      null,      // AdminTrainer instance
    exercise:     null,      // selected exercise
    angleHistory: [],        // for mini chart
    model:        null,
    motion:       null,
    bound:        false,     // prevent duplicate event listeners
};

async function initAdminPage() {
    // Populate exercise dropdown
    if (!S.exercises.length) {
        S.exercises = await API.getExercises() || [];
    }
    const sel = document.getElementById('trainer-exercise-select');
    if (sel) {
        sel.innerHTML = S.exercises.map(ex =>
            `<option value="${ex.id}">${ex.icon} ${ex.name}</option>`
        ).join('');
    }

    // Show baseline chips
    await refreshBaselineChips();

    // Bind buttons once
    if (!AT.bound) {
        AT.bound = true;
        document.getElementById('btn-trainer-begin')?.addEventListener('click', openTrainerCamera);
        document.getElementById('btn-trainer-record')?.addEventListener('click', startTrainerRecording);
        document.getElementById('btn-trainer-stop-rec')?.addEventListener('click', stopTrainerRecording);
        document.getElementById('btn-trainer-back')?.addEventListener('click', resetTrainerToSetup);
        document.getElementById('btn-trainer-save')?.addEventListener('click', saveBaseline);
        document.getElementById('btn-trainer-redo')?.addEventListener('click', resetTrainerToSetup);
    }

    // Show setup panel, hide others
    showTrainerPanel('setup');
}

async function refreshBaselineChips() {
    const list = document.getElementById('trainer-baseline-list');
    if (!list || !S.exercises.length) return;
    const chips = await Promise.all(S.exercises.map(async ex => {
        const m = await API.checkBaseline(ex.id);
        const has = !!m;
        return `<span class="trainer-baseline-chip ${has ? 'has-model' : 'no-model'}">
            ${has ? '✓' : '○'} ${ex.icon} ${ex.name}
            ${has ? `<small>(${m.reps_recorded} reps)</small>` : ''}
        </span>`;
    }));
    list.innerHTML = chips.join('');
}

function showTrainerPanel(which) {
    document.getElementById('trainer-setup').style.display          = which === 'setup'     ? '' : 'none';
    document.getElementById('trainer-recording-panel').style.display = which === 'recording' ? '' : 'none';
    document.getElementById('trainer-results-panel').style.display  = which === 'results'   ? '' : 'none';
}

async function openTrainerCamera() {
    const sel    = document.getElementById('trainer-exercise-select');
    const repInp = document.getElementById('trainer-reps-target');
    if (!sel || !sel.value) return;

    AT.exercise = S.exercises.find(e => e.id === sel.value);
    if (!AT.exercise) return;

    const targetReps = parseInt(repInp?.value || '10', 10);

    // Tear down any previous trainer
    if (AT.trainer) { AT.trainer.stopCamera(); AT.trainer = null; }
    AT.trainer = new AdminTrainer();
    AT.trainer.setup(AT.exercise, targetReps);
    AT.angleHistory = [];

    // Show recording panel
    showTrainerPanel('recording');

    // Fill display labels
    document.getElementById('trainer-exercise-badge').textContent = AT.exercise.name;
    document.getElementById('trainer-rep-target-disp').textContent = targetReps;
    document.getElementById('trainer-primary-joint-disp').textContent =
        (TRAINER_PRIMARY_JOINT[AT.exercise.id] || 'left_knee').replace(/_/g, ' ');

    setTrainerPhase('READY', 'Click Start Recording when positioned');
    document.getElementById('trainer-rep-count').textContent = '0';
    document.getElementById('trainer-angle-val').textContent = '—°';
    document.getElementById('trainer-top-angle').textContent = '—°';
    document.getElementById('trainer-bot-angle').textContent = '—°';
    document.getElementById('trainer-frame-count').textContent = '0';
    document.getElementById('trainer-rep-bar').style.width = '0%';
    document.getElementById('trainer-rec-dot').style.display = 'none';
    document.getElementById('trainer-progress-bar').style.width = '0%';
    document.getElementById('trainer-status-badge').textContent = 'IDLE';
    document.getElementById('trainer-status-badge').className = 'status-badge';

    show('btn-trainer-record'); hide('btn-trainer-stop-rec');
    document.getElementById('trainer-video-loading').style.display = 'flex';

    // Wire callbacks
    AT.trainer.onAngle = (deg) => {
        setText('trainer-angle-val', `${deg}°`);
        setText('trainer-top-angle', `${Math.round(AT.trainer.topAngle)}°`);
        setText('trainer-bot-angle', `${Math.round(AT.trainer.bottomAngle)}°`);
        setText('trainer-frame-count', AT.trainer.motionFrames.length);

        // Live angle chart
        AT.angleHistory.push(deg);
        if (AT.angleHistory.length > 60) AT.angleHistory.shift();
        drawAngleChart();
    };

    AT.trainer.onPhaseChange = (phase) => {
        const label = document.getElementById('trainer-phase-label');
        if (label) {
            label.textContent = phase;
            label.className = 'trainer-phase-label' + (phase === 'BOTTOM' ? ' phase-bottom' : '');
        }
        setText('trainer-phase-val', phase);
    };

    AT.trainer.onRepCount = (count) => {
        const el = document.getElementById('trainer-rep-count');
        if (el) {
            el.textContent = count;
            el.classList.remove('pop');
            void el.offsetWidth;
            el.classList.add('pop');
        }
        const pct = Math.min(count / AT.trainer.targetReps, 1) * 100;
        document.getElementById('trainer-rep-bar').style.width = pct + '%';
        document.getElementById('trainer-progress-bar').style.width = pct + '%';
    };

    AT.trainer.onComplete = (model, motion) => {
        AT.model  = model;
        AT.motion = motion;
        document.getElementById('trainer-rec-dot').style.display = 'none';
        document.getElementById('trainer-status-badge').textContent = 'DONE';
        document.getElementById('trainer-status-badge').className = 'status-badge';
        hide('btn-trainer-stop-rec'); show('btn-trainer-record');
        showTrainerResults(model, motion);
    };

    AT.trainer.onError = (msg) => {
        alert('Trainer error: ' + msg);
    };

    try {
        await AT.trainer.startCamera(
            document.getElementById('trainer-video'),
            document.getElementById('trainer-canvas')
        );

        // Replace fps display id for trainer canvas
        const fpsBadge = document.getElementById('trainer-fps-display');
        if (fpsBadge) {
            // trainer video loading hidden by PoseEngine itself (it checks 'video-loading')
            // Override: manually hide our trainer loading div
            document.getElementById('trainer-video-loading').style.display = 'none';
        }
    } catch (err) {
        document.getElementById('trainer-video-loading').innerHTML = `
            <div style="text-align:center;padding:2rem">
                <div style="font-size:3rem;margin-bottom:1rem">📷</div>
                <h3>Camera Access Required</h3>
                <p style="color:var(--text-secondary);margin:.75rem 0">Allow camera access to use the trainer.</p>
                <button onclick="resetTrainerToSetup()" class="btn btn-glass">← Back</button>
            </div>`;
    }
}

function startTrainerRecording() {
    if (!AT.trainer) return;
    AT.trainer.startRecording();

    document.getElementById('trainer-rec-dot').style.display = 'flex';
    document.getElementById('trainer-status-badge').textContent = 'REC';
    document.getElementById('trainer-status-badge').className = 'status-badge running';
    setTrainerPhase('RECORDING', 'Perform reps at your normal pace');
    document.getElementById('trainer-phase-label').classList.add('phase-recording');

    hide('btn-trainer-record'); show('btn-trainer-stop-rec');
}

function stopTrainerRecording() {
    if (!AT.trainer) return;
    AT.trainer.stopRecording();

    document.getElementById('trainer-rec-dot').style.display = 'none';
    document.getElementById('trainer-status-badge').textContent = 'DONE';
    document.getElementById('trainer-status-badge').className = 'status-badge';
    hide('btn-trainer-stop-rec'); show('btn-trainer-record');
}

function resetTrainerToSetup() {
    if (AT.trainer) { AT.trainer.stopCamera(); AT.trainer = null; }
    AT.model = null; AT.motion = null; AT.angleHistory = [];
    showTrainerPanel('setup');
    refreshBaselineChips();
}

function setTrainerPhase(phase, sub) {
    setText('trainer-phase-label', phase);
    setText('trainer-phase-sub', sub || '');
}

function drawAngleChart() {
    const canvas = document.getElementById('trainer-angle-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const data = AT.angleHistory;

    ctx.clearRect(0,0,w,h);
    if (data.length < 2) return;

    const min = Math.min(...data) - 5;
    const max = Math.max(...data) + 5;
    const rng = max - min || 1;

    ctx.save();
    ctx.strokeStyle = 'rgba(99,102,241,0.8)';
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.shadowColor = 'rgba(99,102,241,0.4)';
    ctx.shadowBlur = 4;

    ctx.beginPath();
    data.forEach((v, i) => {
        const x = (i / (data.length - 1)) * w;
        const y = h - ((v - min) / rng) * h;
        i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();

    // Fill under line
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = 'rgba(99,102,241,0.08)';
    ctx.fill();
    ctx.restore();
}

function showTrainerResults(model, motion) {
    showTrainerPanel('results');

    const primaryJ = TRAINER_PRIMARY_JOINT[model.exercise] || 'left_knee';

    // ── Summary ──────────────────────────────────────────
    const exerciseName = AT.exercise?.name || model.exercise;
    const summary = [
        { k: 'Exercise',       v: exerciseName },
        { k: 'Reps Recorded',  v: model.reps_recorded },
        { k: 'Primary Joint',  v: primaryJ.replace(/_/g,' ') },
        { k: 'Motion Frames',  v: motion.length },
        { k: 'Hold Top',       v: `${model.hold_top_avg_s}s avg` },
        { k: 'Hold Bottom',    v: `${model.hold_bottom_avg_s}s avg` },
    ];
    const primaryStats = model.joints[primaryJ];
    if (primaryStats) {
        summary.push(
            { k: 'Range of Motion', v: `${Math.round(primaryStats.max - primaryStats.min)}°` },
            { k: 'Avg Speed',       v: `${Math.round(primaryStats.speed_avg)}°/s` }
        );
    }
    document.getElementById('trainer-summary-list').innerHTML = summary.map(s =>
        `<div class="trainer-summary-item">
            <span class="trainer-summary-key">${s.k}</span>
            <span class="trainer-summary-val">${s.v}</span>
        </div>`
    ).join('');

    // Sub text
    setText('trainer-results-sub',
        `${model.reps_recorded} reps · ${motion.length} frames recorded · ready to apply`);

    // ── Stats table ──────────────────────────────────────
    const joints = Object.entries(model.joints);
    const tableHtml = `
        <div class="trainer-stats-row header">
            <span>Joint</span>
            <span style="text-align:right">Min</span>
            <span style="text-align:right">Max</span>
            <span style="text-align:right">Avg</span>
            <span style="text-align:right">Std</span>
            <span style="text-align:right">Speed</span>
        </div>` +
        joints.map(([jName, st]) => {
            const isPrimary = jName === primaryJ;
            return `<div class="trainer-stats-row">
                <span class="joint-name ${isPrimary ? 'primary' : ''}">${jName.replace(/_/g,' ')}</span>
                <span class="stat-val">${Math.round(st.min)}°</span>
                <span class="stat-val">${Math.round(st.max)}°</span>
                <span class="stat-val">${Math.round(st.avg)}°</span>
                <span class="stat-val">${Math.round(st.std)}°</span>
                <span class="stat-val speed">${Math.round(st.speed_avg)}°/s</span>
            </div>`;
        }).join('');
    document.getElementById('trainer-stats-table').innerHTML = tableHtml;

    // Reset save status
    const statusEl = document.getElementById('trainer-save-status');
    statusEl.style.display = 'none';
    statusEl.className = 'trainer-save-status';
}

async function saveBaseline() {
    if (!AT.model || !AT.motion) return;
    const btn = document.getElementById('btn-trainer-save');
    const statusEl = document.getElementById('trainer-save-status');

    btn.disabled = true;
    btn.textContent = '⏳ Saving…';

    try {
        const result = await API.saveBaseline(AT.model.exercise, AT.model, AT.motion);
        if (result?.status === 'saved') {
            statusEl.textContent = `✅ Saved! ${result.reps} reps baseline for ${AT.exercise?.name}. Reload coach to use it.`;
            statusEl.className = 'trainer-save-status success';
            statusEl.style.display = 'block';
            btn.textContent = '✅ Saved!';
            // Refresh chips
            await refreshBaselineChips();
            // Invalidate cached model so next coach session fetches fresh
            S.model = null;
        } else {
            throw new Error(result?.error || 'Unknown error');
        }
    } catch (err) {
        statusEl.textContent = '❌ Save failed: ' + err.message;
        statusEl.className = 'trainer-save-status error';
        statusEl.style.display = 'block';
        btn.disabled = false;
        btn.textContent = '💾 Save Baseline';
    }
}

// ═══════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
function setClass(id, cls) { const el = document.getElementById(id); if (el) el.className = (el.className.split(' ')[0] + ' ' + cls).trim(); }
function show(id) { const el = document.getElementById(id); if (el) el.style.display = ''; }
function hide(id) { const el = document.getElementById(id); if (el) el.style.display = 'none'; }
function showElement(id) { const el = document.getElementById(id); if (el) el.style.display = 'flex'; }
function hideElement(id) { const el = document.getElementById(id); if (el) el.style.display = 'none'; }

// ─── Audio button state helper ────────────────────────────────────────────────
function _setAudioBtnState(on) {
    const btn   = document.getElementById('btn-toggle-audio');
    const icon  = document.getElementById('audio-icon');
    const label = document.getElementById('audio-label');
    if (!btn) return;
    if (icon)  icon.textContent  = on ? '🔊' : '🔇';
    if (label) label.textContent = on ? 'Voice' : 'Muted';
    btn.classList.toggle('audio-muted', !on);
}

// ═══════════════════════════════════════════════════════
//  EVENT LISTENERS
// ═══════════════════════════════════════════════════════
function bindEvents() {
    window.addEventListener('hashchange', handleHash);

    document.getElementById('btn-start-session')?.addEventListener('click', startSession);
    document.getElementById('btn-stop-session')?.addEventListener('click',  stopSession);

    document.getElementById('btn-toggle-audio')?.addEventListener('click', () => {
        const on = S.feedback.toggleAudio();
        _setAudioBtnState(on);
    });

    document.getElementById('btn-replay-demo')?.addEventListener('click', () => {
        if (!S.isReplaying && S.motionData) {
            triggerAutoReplay('Manual replay requested');
        }
    });

    // Modal buttons
    document.getElementById('modal-restart')?.addEventListener('click', () => {
        document.getElementById('session-modal').style.display = 'none';
        if (S.exercise) initCoachPage(S.exercise.id);
    });
    document.getElementById('modal-dashboard')?.addEventListener('click', () => {
        document.getElementById('session-modal').style.display = 'none';
    });
    document.getElementById('session-modal')?.addEventListener('click', (e) => {
        if (e.target === e.currentTarget) e.currentTarget.style.display = 'none';
    });

    // Keyboard shortcuts (mirrors coach.py key handling)
    window.addEventListener('keydown', (e) => {
        if (S.page !== 'coach') return;
        if (e.key === 's' || e.key === 'S') { if (!S.session?.isRunning) startSession(); }
        if (e.key === 'e' || e.key === 'E') { if (S.session?.isRunning)  stopSession();  }
        if (e.key === 'v' || e.key === 'V') { document.getElementById('btn-toggle-audio')?.click(); }
        if (e.key === 'r' || e.key === 'R') { document.getElementById('btn-replay-demo')?.click(); }
    });

    // Cleanup when leaving coach or admin page
    window.addEventListener('hashchange', () => {
        if (S.page === 'coach' && !location.hash.includes('coach')) {
            if (S.session?.isRunning) stopSession();
            if (S.poseEngine) { S.poseEngine.stop(); S.poseEngine = null; }
            stopAvatarRef();
        }
        if (S.page === 'admin' && !location.hash.includes('admin')) {
            if (AT.trainer) { AT.trainer.stopCamera(); AT.trainer = null; }
        }
    });
}

// Expose for inline onclick in error HTML
window.resetTrainerToSetup = resetTrainerToSetup;

// ═══════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════
function init() {
    initParticles();
    initHeroSkeleton();
    animateStatCounters();
    bindEvents();
    handleHash();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

})();
