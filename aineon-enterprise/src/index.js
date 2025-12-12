const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;
const HOST = '0.0.0.0';

app.get('/', (req, res) => {
  res.json({
    message: 'AINEON Enterprise Platform API',
    status: 'running',
    timestamp: new Date().toISOString(),
    port: PORT
  });
});

app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'aineon-enterprise'
  });
});

app.listen(PORT, HOST, () => {
  console.log(`íº€ Server running on http://${HOST}:${PORT}`);
  console.log(`âœ… Health check: http://${HOST}:${PORT}/health`);
});
