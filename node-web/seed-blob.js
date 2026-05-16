/**
 * Seed Vercel Blob with all baseline model and motion files.
 * Run once: BLOB_READ_WRITE_TOKEN=<token> node seed-blob.js
 */

const { put } = require('@vercel/blob');
const fs   = require('fs');
const path = require('path');

const DIR = path.join(__dirname, '..', 'baselines');

async function seed() {
    const files = fs.readdirSync(DIR).filter(f => f.endsWith('.json'));
    console.log(`Seeding ${files.length} files to Vercel Blob...\n`);

    for (const file of files) {
        const filepath = path.join(DIR, file);
        const content  = fs.readFileSync(filepath);
        const isModel  = file.endsWith('_model.json');
        const blobPath = isModel
            ? `models/${file}`
            : `motions/${file}`;

        const result = await put(blobPath, content, {
            access:           'public',
            contentType:      'application/json',
            addRandomSuffix:  false,
        });
        const kb = (content.length / 1024).toFixed(0);
        console.log(`  ✓ ${blobPath}  (${kb} KB)  → ${result.url}`);
    }

    console.log('\nSeed complete.');
}

seed().catch(err => { console.error(err); process.exit(1); });
