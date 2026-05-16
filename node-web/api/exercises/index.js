const EXERCISES = require('../../lib/exercises');

module.exports = (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.json(Object.values(EXERCISES));
};
