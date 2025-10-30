const express = require('express');
const cors = require('cors');
const WebSocket = require('ws');
const net = require('net');

// Function to find available port
function findAvailablePort(startPort) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.listen(startPort, () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
    server.on('error', () => {
      resolve(findAvailablePort(startPort + 1));
    });
  });
}

async function startServer() {
  // Find available ports
  const API_PORT = await findAvailablePort(5000);
  const WS_PORT = await findAvailablePort(API_PORT + 1);

  const app = express();
  
  console.log('í¾¯ MASTER DASHBOARD - AUTO PORT ASSIGNMENT');
  console.log('=========================================');

  app.use(cors());
  app.use(express.json());

  const masterDashboardData = {
    system: {
      name: "AINEON MASTER DASHBOARD",
      version: "2.0.0",
      status: "í¿¢ OPERATIONAL",
      aiMode: "AUTONOMOUS",
      uptime: "99.7%",
      lastUpdate: new Date().toISOString(),
      ports: {
        api: API_PORT,
        websocket: WS_PORT
      }
    },
    wallet: {
      totalBalance: 12847290,
      activeThreats: 0,
      multiSigStatus: '2/3 Ready',
      status: 'SECURE'
    },
    trading: {
      successRate: 98.7,
      latency: 450,
      profitVsTarget: 50,
      mode: 'AI Optimized'
    },
    optimization: {
      activeStrategies: 15,
      inResearch: 8,
      pipelineHealth: 'Optimal'
    },
    monitoring: {
      botArmy: {
        seekers: { active: 42, total: 50, accuracy: 96.2 },
        captain: { active: 1, total: 1, accuracy: 98.7 }
      }
    },
    ai: {
      patternAccuracy: 96.2,
      confidenceScore: 94.8,
      activePatterns: 12,
      modelHealth: 'Strong'
    },
    profit: {
      totalProfits: 2147380,
      taxSaved: 12400,
      compliance: 100,
      status: 'Optimal'
    },
    flashloan: {
      utilization: 68,
      currentROI: 3.2,
      providers: '4/4 Healthy',
      status: 'Active'
    },
    security: {
      activeThreats: 0,
      threatLevel: 'LOW'
    },
    health: {
      systemHealth: 'Optimal',
      uptime: 99.4
    }
  };

  app.get('/api/metrics', (req, res) => {
    console.log('í³Š Metrics requested');
    masterDashboardData.system.lastUpdate = new Date().toISOString();
    res.json(masterDashboardData);
  });

  app.post('/api/ai/autonomy/enable', (req, res) => {
    const { mode } = req.body;
    console.log(`í·  AI mode: ${mode}`);
    masterDashboardData.system.aiMode = mode;
    res.json({ success: true, message: `AI mode set to ${mode}` });
  });

  app.get('/health', (req, res) => {
    res.json({ 
      status: 'OK', 
      service: 'AINEON MASTER DASHBOARD',
      ports: { api: API_PORT, websocket: WS_PORT }
    });
  });

  app.get('/', (req, res) => {
    res.json({
      message: 'íº€ AINEON MASTER DASHBOARD - AUTO PORTS',
      endpoints: {
        metrics: `GET http://localhost:${API_PORT}/api/metrics`,
        aiControl: `POST http://localhost:${API_PORT}/api/ai/autonomy/enable`,
        health: `GET http://localhost:${API_PORT}/health`
      },
      ports: {
        api: API_PORT,
        websocket: WS_PORT
      }
    });
  });

  // Start HTTP server
  const server = app.listen(API_PORT, () => {
    console.log(`íº€ API Server: http://localhost:${API_PORT}`);
    console.log(`í³Š Dashboard: http://localhost:${API_PORT}/`);
    console.log(`í³ˆ Metrics: http://localhost:${API_PORT}/api/metrics`);
    console.log(`â¤ï¸  Health: http://localhost:${API_PORT}/health`);
  });

  // Start WebSocket server
  const wss = new WebSocket.Server({ port: WS_PORT });
  console.log(`í´Œ WebSocket: ws://localhost:${WS_PORT}`);
  console.log('âœ… MASTER DASHBOARD READY WITH AUTO PORTS');

  // Write ports to file for frontend
  const fs = require('fs');
  fs.writeFileSync('current-ports.json', JSON.stringify({
    api: API_PORT,
    websocket: WS_PORT
  }));

  process.on('SIGINT', () => {
    console.log('\ní»‘ Shutting down...');
    server.close(() => process.exit(0));
  });
}

startServer().catch(console.error);
