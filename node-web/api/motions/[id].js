const { list } = require('@vercel/blob');
const EXERCISES = require('../../lib/exercises');

module.exports = async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });

    const { id } = req.query;
    if (!EXERCISES[id]) return res.status(404).json({ error: 'Exercise not found' });

    try {
        const { blobs } = await list({ prefix: `motions/${id}_motion.json`, limit: 1 });
        if (!blobs.length) return res.status(404).json({ error: 'Motion data not found' });

        const resp = await fetch(blobs[0].url);
        if (!resp.ok) return res.status(404).json({ error: 'Motion data not found' });
        const data = await resp.json();
        res.json(data);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
};
