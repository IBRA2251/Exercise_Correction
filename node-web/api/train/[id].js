const { put } = require('@vercel/blob');
const EXERCISES = require('../../lib/exercises');

module.exports = async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST,OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    if (req.method === 'OPTIONS') return res.status(200).end();
    if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

    const { id } = req.query;
    if (!EXERCISES[id]) return res.status(404).json({ error: 'Exercise not found' });

    const { model, motion } = req.body || {};
    if (!model || !motion) return res.status(400).json({ error: 'model and motion required' });

    try {
        await put(`models/${id}_model.json`, JSON.stringify(model), {
            access: 'public',
            contentType: 'application/json',
            addRandomSuffix: false,
            allowOverwrite: true,
        });

        await put(`motions/${id}_motion.json`, JSON.stringify(motion), {
            access: 'public',
            contentType: 'application/json',
            addRandomSuffix: false,
            allowOverwrite: true,
        });

        res.json({ status: 'saved', exercise: id, reps: model.reps_recorded });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
};
