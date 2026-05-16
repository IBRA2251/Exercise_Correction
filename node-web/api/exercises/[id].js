const EXERCISES = require('../../lib/exercises');

module.exports = (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    const ex = EXERCISES[req.query.id];
    if (!ex) return res.status(404).json({ error: 'Exercise not found' });
    res.json(ex);
};
