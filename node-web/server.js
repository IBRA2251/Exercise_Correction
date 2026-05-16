/**
 * FormAI — Node.js/Express Backend
 * Replaces Flask web/server.py with identical API surface.
 * All pose estimation, FSM, and feedback runs in the browser.
 */

const express = require('express');
const path    = require('path');
const fs      = require('fs');
const cors    = require('cors');

const app  = express();
const PORT = process.env.PORT || 3000;

const PROJECT_ROOT  = path.join(__dirname, '..');
const BASELINES_DIR = path.join(PROJECT_ROOT, 'baselines');
const SESSIONS_FILE = path.join(__dirname, 'sessions.json');
const PUBLIC_DIR    = path.join(__dirname, 'public');

app.use(cors());
app.use(express.json());
app.use(express.static(PUBLIC_DIR));

// ─── Exercise Catalog (mirrors config.py + web/server.py) ───────────────────
const EXERCISE_CATALOG = {
    squat: {
        id: 'squat',
        name: 'Squat',
        type: 'single',
        primary_angle: 'left_knee',
        direction: 'decrease_then_increase',
        min_rom: 35,
        difficulty: 'Beginner',
        category: 'Strength',
        muscles: ['Quadriceps', 'Glutes', 'Hamstrings', 'Core'],
        description: 'A fundamental lower-body exercise that targets your quads, glutes, and core stability.',
        instructions: 'Stand with feet shoulder-width apart, lower your hips back and down as if sitting in a chair, then push back up through your heels.',
        icon: '🦵',
        color: '#6366f1',
    },
    pushup: {
        id: 'pushup',
        name: 'Push-Up',
        type: 'single',
        primary_angle: 'left_elbow',
        direction: 'decrease_then_increase',
        min_rom: 30,
        difficulty: 'Beginner',
        category: 'Strength',
        muscles: ['Chest', 'Triceps', 'Shoulders', 'Core'],
        description: 'The classic upper-body compound movement that builds chest, tricep, and shoulder strength.',
        instructions: 'Start in a plank position with hands shoulder-width apart. Lower your chest to the floor then push back up with full arm extension.',
        icon: '💪',
        color: '#8b5cf6',
    },
    bicep_curl: {
        id: 'bicep_curl',
        name: 'Bicep Curl',
        type: 'single',
        primary_angle: 'left_elbow',
        direction: 'decrease_then_increase',
        min_rom: 40,
        difficulty: 'Beginner',
        category: 'Strength',
        muscles: ['Biceps', 'Forearms', 'Brachialis'],
        description: 'An isolation exercise that targets the biceps for arm strength and definition.',
        instructions: 'Stand with arms at your sides, curl your forearms up to shoulder level by bending at the elbows, then lower slowly.',
        icon: '🏋️',
        color: '#a855f7',
    },
    boxing: {
        id: 'boxing',
        name: 'Boxing',
        type: 'boxing',
        primary_angle: 'right_elbow',
        direction: '',
        min_rom: 0,
        difficulty: 'Intermediate',
        category: 'Combat',
        muscles: ['Shoulders', 'Core', 'Arms', 'Legs'],
        description: 'High-intensity boxing drills that improve speed, coordination, and cardiovascular fitness.',
        instructions: 'Maintain a guard stance with hands at chin level, throw punches with speed and full extension, return to guard between punches.',
        icon: '🥊',
        color: '#f43f5e',
    },
    yoga: {
        id: 'yoga',
        name: 'Yoga',
        type: 'yoga',
        primary_angle: 'left_knee',
        direction: '',
        min_rom: 0,
        difficulty: 'Intermediate',
        category: 'Flexibility',
        muscles: ['Full Body', 'Core', 'Balance', 'Flexibility'],
        description: 'Yoga poses that improve flexibility, balance, and mindfulness through controlled, stable movements.',
        instructions: 'Move into the pose slowly and with control, breathe steadily, hold the position with stability, then release gently.',
        icon: '🧘',
        color: '#06b6d4',
    },
};

// ─── Session helpers ─────────────────────────────────────────────────────────
function loadSessions() {
    try {
        if (fs.existsSync(SESSIONS_FILE)) {
            return JSON.parse(fs.readFileSync(SESSIONS_FILE, 'utf8'));
        }
    } catch { /* ignore */ }
    return [];
}

function saveSessions(sessions) {
    fs.writeFileSync(SESSIONS_FILE, JSON.stringify(sessions, null, 2));
}

// ─── API Routes ──────────────────────────────────────────────────────────────

app.get('/api/exercises', (req, res) => {
    res.json(Object.values(EXERCISE_CATALOG));
});

app.get('/api/exercises/:id', (req, res) => {
    const ex = EXERCISE_CATALOG[req.params.id];
    if (!ex) return res.status(404).json({ error: 'Exercise not found' });
    res.json(ex);
});

app.get('/api/models/:id', (req, res) => {
    const p = path.join(BASELINES_DIR, `${req.params.id}_model.json`);
    if (!fs.existsSync(p)) return res.status(404).json({ error: 'Model not found' });
    const model = JSON.parse(fs.readFileSync(p, 'utf8'));
    delete model.fsm_stages; // strip large array from response
    res.json(model);
});

app.get('/api/motions/:id', (req, res) => {
    const p = path.join(BASELINES_DIR, `${req.params.id}_motion.json`);
    if (!fs.existsSync(p)) return res.status(404).json({ error: 'Motion data not found' });
    res.json(JSON.parse(fs.readFileSync(p, 'utf8')));
});

app.get('/api/sessions', (req, res) => {
    res.json(loadSessions());
});

app.post('/api/sessions', (req, res) => {
    const data = req.body;
    if (!data) return res.status(400).json({ error: 'No data provided' });

    data.id        = Date.now();
    data.timestamp = new Date().toISOString().replace('T', ' ').substring(0, 19);

    const sessions = loadSessions();
    sessions.unshift(data);
    saveSessions(sessions.slice(0, 100)); // keep last 100

    res.status(201).json({ status: 'saved', id: data.id });
});

// ─── Admin: save trained baseline ────────────────────────────────────────────
app.post('/api/train/:id', (req, res) => {
    const id = req.params.id;
    if (!EXERCISE_CATALOG[id]) return res.status(404).json({ error: 'Exercise not found' });

    const { model, motion } = req.body;
    if (!model || !motion) return res.status(400).json({ error: 'model and motion required' });

    try {
        if (!fs.existsSync(BASELINES_DIR)) fs.mkdirSync(BASELINES_DIR, { recursive: true });
        fs.writeFileSync(path.join(BASELINES_DIR, `${id}_model.json`),  JSON.stringify(model,  null, 2));
        fs.writeFileSync(path.join(BASELINES_DIR, `${id}_motion.json`), JSON.stringify(motion, null, 2));
        res.json({ status: 'saved', exercise: id, reps: model.reps_recorded });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// ─── SPA fallback ────────────────────────────────────────────────────────────
app.get('*', (req, res) => {
    res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

// ─── Start ───────────────────────────────────────────────────────────────────
if (process.env.VERCEL !== '1') {
    app.listen(PORT, () => {
        console.log('\n  ⚡ FormAI — AI Exercise Coach');
        console.log(`  Open http://localhost:${PORT} in your browser\n`);
    });
}
module.exports = app;
