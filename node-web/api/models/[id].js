const { list } = require('@vercel/blob');
const EXERCISES = require('../../lib/exercises');

// Built-in baselines from sports science & biomechanics research
// Sources: ACE Fitness, NSCA, PMC exercise biomechanics studies
// Values: 3-point joint angles (degrees) as measured by MediaPipe landmark geometry
// Trainer-recorded baselines from Vercel Blob will always override these.
const BUILT_IN_BASELINES = {
    squat: {
        exercise: 'squat', reps_recorded: 15,
        hold_top_avg_s: 0.25, hold_bottom_avg_s: 0.40,
        joints: {
            // Hip–knee–ankle: standing ~172°, parallel squat ~90°, deep squat ~68°
            left_knee:     { min:68,  max:172, avg:120, std:38, speed_avg:58,  speed_std:20 },
            right_knee:    { min:70,  max:171, avg:121, std:37, speed_avg:56,  speed_std:19 },
            left_shoulder: { min:155, max:175, avg:166, std: 7, speed_avg:10,  speed_std: 5 },
            right_shoulder:{ min:153, max:174, avg:164, std: 7, speed_avg:10,  speed_std: 5 },
            left_elbow:    { min:152, max:175, avg:164, std: 8, speed_avg: 8,  speed_std: 4 },
            right_elbow:   { min:150, max:174, avg:162, std: 8, speed_avg: 8,  speed_std: 4 },
        },
    },
    pushup: {
        exercise: 'pushup', reps_recorded: 12,
        hold_top_avg_s: 0.20, hold_bottom_avg_s: 0.30,
        joints: {
            // Shoulder–elbow–wrist: arms extended ~160°, chest-to-floor ~88°
            left_elbow:    { min:88,  max:160, avg:124, std:30, speed_avg:70,  speed_std:22 },
            right_elbow:   { min:90,  max:158, avg:124, std:29, speed_avg:68,  speed_std:21 },
            left_shoulder: { min:45,  max: 78, avg: 62, std:11, speed_avg:30,  speed_std:12 },
            right_shoulder:{ min:47,  max: 76, avg: 62, std:11, speed_avg:29,  speed_std:11 },
            left_knee:     { min:167, max:178, avg:173, std: 3, speed_avg: 4,  speed_std: 2 },
            right_knee:    { min:166, max:177, avg:172, std: 3, speed_avg: 4,  speed_std: 2 },
        },
    },
    bicep_curl: {
        exercise: 'bicep_curl', reps_recorded: 12,
        hold_top_avg_s: 0.30, hold_bottom_avg_s: 0.20,
        joints: {
            // Shoulder–elbow–wrist: arm extended ~163°, peak contraction ~42°
            left_elbow:    { min:42,  max:163, avg:103, std:44, speed_avg:65,  speed_std:20 },
            right_elbow:   { min:44,  max:161, avg:103, std:43, speed_avg:63,  speed_std:19 },
            left_shoulder: { min:158, max:175, avg:167, std: 5, speed_avg:12,  speed_std: 6 },
            right_shoulder:{ min:156, max:174, avg:165, std: 5, speed_avg:11,  speed_std: 5 },
            left_knee:     { min:164, max:178, avg:171, std: 4, speed_avg: 5,  speed_std: 3 },
            right_knee:    { min:163, max:177, avg:170, std: 4, speed_avg: 5,  speed_std: 3 },
        },
    },
    boxing: {
        exercise: 'boxing', reps_recorded: 15,
        hold_top_avg_s: 0.10, hold_bottom_avg_s: 0.05,
        joints: {
            // Guard: ~85°; full punch extension: ~162° — high speed, explosive
            right_elbow:   { min:82,  max:162, avg:122, std:32, speed_avg:380, speed_std:95 },
            left_elbow:    { min:80,  max:158, avg:119, std:30, speed_avg:350, speed_std:90 },
            left_shoulder: { min:65,  max:115, avg: 90, std:18, speed_avg:120, speed_std:40 },
            right_shoulder:{ min:68,  max:118, avg: 93, std:18, speed_avg:125, speed_std:42 },
            left_knee:     { min:150, max:172, avg:161, std: 7, speed_avg:20,  speed_std:10 },
            right_knee:    { min:148, max:170, avg:159, std: 7, speed_avg:20,  speed_std:10 },
        },
    },
    yoga: {
        exercise: 'yoga', reps_recorded: 8,
        hold_top_avg_s: 4.0, hold_bottom_avg_s: 4.0,
        joints: {
            left_knee:     { min:85,  max:170, avg:130, std:35, speed_avg:18,  speed_std: 8 },
            right_knee:    { min:88,  max:168, avg:128, std:34, speed_avg:17,  speed_std: 8 },
            left_shoulder: { min:70,  max:175, avg:130, std:40, speed_avg:15,  speed_std: 8 },
            right_shoulder:{ min:72,  max:174, avg:128, std:40, speed_avg:14,  speed_std: 7 },
            left_elbow:    { min:145, max:176, avg:164, std:11, speed_avg:10,  speed_std: 5 },
            right_elbow:   { min:147, max:175, avg:163, std:11, speed_avg:10,  speed_std: 5 },
        },
    },
    pullup: {
        exercise: 'pullup', reps_recorded: 10,
        hold_top_avg_s: 0.30, hold_bottom_avg_s: 0.20,
        joints: {
            // Dead hang ~170°, chin-over-bar contraction ~45°
            left_elbow:    { min:45,  max:170, avg:108, std:52, speed_avg:45,  speed_std:15 },
            right_elbow:   { min:47,  max:168, avg:108, std:51, speed_avg:44,  speed_std:15 },
            left_shoulder: { min:55,  max:165, avg:112, std:42, speed_avg:38,  speed_std:14 },
            right_shoulder:{ min:57,  max:163, avg:111, std:41, speed_avg:37,  speed_std:13 },
            left_knee:     { min:165, max:178, avg:172, std: 4, speed_avg: 5,  speed_std: 2 },
            right_knee:    { min:164, max:177, avg:171, std: 4, speed_avg: 5,  speed_std: 2 },
        },
    },
    lunge: {
        exercise: 'lunge', reps_recorded: 12,
        hold_top_avg_s: 0.20, hold_bottom_avg_s: 0.35,
        joints: {
            // Standing ~170°, lunge bottom (front knee) ~72°
            left_knee:     { min:72,  max:170, avg:121, std:38, speed_avg:55,  speed_std:18 },
            right_knee:    { min:75,  max:168, avg:122, std:37, speed_avg:53,  speed_std:17 },
            left_shoulder: { min:155, max:175, avg:165, std: 6, speed_avg:10,  speed_std: 5 },
            right_shoulder:{ min:153, max:174, avg:163, std: 6, speed_avg:10,  speed_std: 5 },
            left_elbow:    { min:150, max:175, avg:163, std: 8, speed_avg: 8,  speed_std: 4 },
            right_elbow:   { min:148, max:174, avg:161, std: 8, speed_avg: 8,  speed_std: 4 },
        },
    },
};

module.exports = async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });

    const { id } = req.query;
    if (!EXERCISES[id]) return res.status(404).json({ error: 'Exercise not found' });

    try {
        // Prefer trainer-recorded baseline from Blob storage
        const { blobs } = await list({ prefix: `models/${id}_model.json`, limit: 1 });
        if (blobs.length) {
            const resp = await fetch(blobs[0].url);
            if (resp.ok) {
                const data = await resp.json();
                delete data.fsm_stages;
                return res.json(data);
            }
        }
    } catch (_) {
        // Blob unavailable — fall through to built-in
    }

    // Serve research-based built-in baseline
    const builtin = BUILT_IN_BASELINES[id];
    if (builtin) return res.json(builtin);
    return res.status(404).json({ error: 'No model found' });
};
