const { put, list } = require('@vercel/blob');

const SESSIONS_BLOB = 'data/sessions.json';

async function readSessions() {
    try {
        const { blobs } = await list({ prefix: SESSIONS_BLOB, limit: 1 });
        if (!blobs.length) return [];
        const resp = await fetch(blobs[0].url);
        if (!resp.ok) return [];
        return await resp.json();
    } catch {
        return [];
    }
}

module.exports = async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    if (req.method === 'OPTIONS') return res.status(200).end();

    try {
        if (req.method === 'GET') {
            const sessions = await readSessions();
            return res.json(sessions);
        }

        if (req.method === 'POST') {
            const data = req.body || {};
            data.id        = Date.now();
            data.timestamp = new Date().toISOString().replace('T', ' ').substring(0, 19);

            const sessions = await readSessions();
            sessions.unshift(data);
            const trimmed = sessions.slice(0, 100);

            await put(SESSIONS_BLOB, JSON.stringify(trimmed), {
                access: 'public',
                contentType: 'application/json',
                addRandomSuffix: false,
            });

            return res.status(201).json({ status: 'saved', id: data.id });
        }

        res.status(405).json({ error: 'Method not allowed' });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
};
