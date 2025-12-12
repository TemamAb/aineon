const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

// Health check endpoint (required for Render)
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    service: 'aineon-flashloan',
    timestamp: new Date().toISOString()
  });
});

// Main endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'AINEON Flashloan Platform',
    status: 'running',
    version: '1.0.0',
    endpoints: {
      health: '/health',
      api: '/api/v1 (coming soon)'
    }
  });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Ì∫Ä AINEON Flashloan Server started`);
  console.log(`Ì≥° Port: ${PORT}`);
  console.log(`Ìºê URL: http://0.0.0.0:${PORT}`);
});
