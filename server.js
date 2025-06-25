const express = require('express');
const multer = require('multer');
const path = require('path');
const { exec } = require('child_process');
const app = express();
const PORT = 3000;
const cors = require('cors');
app.use(cors());

// Add static file serving (after middleware setup)
app.use(express.static(path.join(__dirname, 'public')));

// Serve index.html for all non-API, non-static routes (for SPA and direct URL access)
app.get(/^\/(?!predict|uploads|static|api).*/, (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const upload = multer({
  dest: 'uploads/',
  fileFilter: function (req, file, cb) {
    const filetypes = /jpeg|jpg|png/;
    const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
    if (extname) return cb(null, true);
    cb(new Error('Only images are allowed'));
  }
});

app.post('/predict', upload.single('image'), (req, res) => {
  if (!req.file) {
    console.error('[server.js] No image uploaded.');
    return res.status(400).json({ error: 'No image uploaded.' });
  }

  const imagePath = path.join(__dirname, req.file.path);
  const pythonScript = 'predict.py';

  const command = `python ${pythonScript} "${imagePath}"`;
  console.log(`[server.js] Running command: ${command}`);

  exec(command, (error, stdout, stderr) => {
    console.log('--- predict.py STDOUT ---');
    console.log(stdout);
    console.log('--- predict.py STDERR ---');
    console.log(stderr);
    if (error) {
      console.error(`[server.js] Error: ${error}`);
      return res.status(500).json({ error: 'Model prediction failed.' });
    }
    // Remove TensorFlow logs and find the last JSON line
    const lines = stdout.trim().split(/\r?\n/);
    let jsonLine = null;
    for (let i = lines.length - 1; i >= 0; i--) {
      try {
        const parsed = JSON.parse(lines[i]);
        jsonLine = parsed;
        break;
      } catch (e) { continue; }
    }
    if (!jsonLine) {
      console.error('[server.js] No valid JSON output from predict.py');
      return res.status(500).json({ error: 'Invalid model output.' });
    }
    if (jsonLine.error) {
      console.error(`[server.js] Model error: ${jsonLine.error}`);
      return res.status(500).json({ error: jsonLine.error });
    }
    res.json({ label: jsonLine.label, confidence: jsonLine.confidence });
  });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
