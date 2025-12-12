const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

// Bind to 0.0.0.0 for Docker compatibility
const HOST = '0.0.0.0';

// Health check endpoint (required for Docker healthcheck)
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'aineon-enterprise',
    environment: process.env.NODE_ENV || 'production'
  });
});

// Main endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'AINEON Enterprise Platform API',
    status: 'running',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    port: PORT,
    deployment: 'docker'
  });
});

// Start server
app.listen(PORT, HOST, () => {
  console.log(`Ì∫Ä AINEON Server running in Docker`);
  console.log(`Ì≥° Host: ${HOST}, Port: ${PORT}`);
  console.log(`‚úÖ Health endpoint: http://${HOST}:${PORT}/health`);
  console.log(`Ìºê Main endpoint: http://${HOST}:${PORT}/`);
});
