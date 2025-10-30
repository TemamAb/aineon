const express = require('express');
const path = require('path');
const app = express();

// Serve static dashboard files
app.use(express.static('.'));

// API proxy to backend
app.use('/api', (req, res) => {
  // Proxy API calls to backend server
  require('http').request({
    hostname: 'localhost',
    port: 3000,
    path: req.url,
    method: req.method,
    headers: req.headers
  }, (backendRes) => {
    res.writeHead(backendRes.statusCode, backendRes.headers);
    backendRes.pipe(res);
  }).end();
});

// Serve main dashboard page
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'dashboard.html'));
});

app.listen(8080, () => {
  console.log('í³Š Dashboard frontend on http://localhost:8080');
});
