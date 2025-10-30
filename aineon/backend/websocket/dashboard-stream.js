const WebSocket = require('ws');

class DashboardWebSocketServer {
  constructor(port = 8081) {
    this.port = port;
    this.wss = null;
    this.clients = new Set();
  }

  start() {
    this.wss = new WebSocket.Server({ port: this.port });
    
    this.wss.on('connection', (ws) => {
      console.log('í´Œ New dashboard WebSocket connection');
      this.clients.add(ws);
      
      // Send initial connection confirmation
      ws.send(JSON.stringify({
        type: 'CONNECTION_ESTABLISHED',
        payload: { 
          message: 'Connected to AINEON Dashboard Stream',
          services: ['trading', 'ai', 'security', 'health'],
          timestamp: new Date().toISOString()
        },
        timestamp: Date.now()
      }));

      // Start real-time data streaming from all services
      this.startRealTimeStreaming(ws);

      ws.on('close', () => {
        console.log('í´Œ Dashboard WebSocket disconnected');
        this.clients.delete(ws);
      });

      ws.on('error', (error) => {
        console.error('Dashboard WebSocket error:', error);
        this.clients.delete(ws);
      });
    });

    console.log(`í³¡ Dashboard WebSocket server running on port ${this.port}`);
  }

  startRealTimeStreaming(ws) {
    // Trading data stream (every 5 seconds)
    const tradingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        this.streamTradingData(ws);
      }
    }, 5000);

    // AI intelligence stream (every 10 seconds)
    const aiInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        this.streamAIData(ws);
      }
    }, 10000);

    // Security events stream (real-time)
    const securityInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        this.streamSecurityData(ws);
      }
    }, 2000);

    // System health stream (every 5 seconds)
    const healthInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        this.streamHealthData(ws);
      }
    }, 5000);

    // Clean up intervals when connection closes
    ws.on('close', () => {
      clearInterval(tradingInterval);
      clearInterval(aiInterval);
      clearInterval(securityInterval);
      clearInterval(healthInterval);
    });
  }

  async streamTradingData(ws) {
    try {
      const tradingData = await require('../performance-monitor').getLiveMetrics();
      const message = {
        type: 'TRADING_UPDATE',
        payload: tradingData,
        timestamp: Date.now()
      };
      ws.send(JSON.stringify(message));
    } catch (error) {
      console.error('Failed to stream trading data:', error);
    }
  }

  async streamAIData(ws) {
    try {
      const aiData = await require('../pattern-analyzer').getIntelligence();
      const message = {
        type: 'AI_UPDATE',
        payload: aiData,
        timestamp: Date.now()
      };
      ws.send(JSON.stringify(message));
    } catch (error) {
      console.error('Failed to stream AI data:', error);
    }
  }

  async streamSecurityData(ws) {
    try {
      const securityData = await require('../enhanced-security').getThreatStatus();
      const message = {
        type: 'SECURITY_UPDATE',
        payload: securityData,
        timestamp: Date.now()
      };
      ws.send(JSON.stringify(message));
    } catch (error) {
      console.error('Failed to stream security data:', error);
    }
  }

  async streamHealthData(ws) {
    try {
      const healthData = await require('../health-monitor').getSystemStatus();
      const message = {
        type: 'HEALTH_UPDATE',
        payload: healthData,
        timestamp: Date.now()
      };
      ws.send(JSON.stringify(message));
    } catch (error) {
      console.error('Failed to stream health data:', error);
    }
  }

  broadcastToAll(message) {
    this.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(message));
      }
    });
  }

  stop() {
    if (this.wss) {
      this.wss.close();
      console.log('Dashboard WebSocket server stopped');
    }
  }
}

module.exports = DashboardWebSocketServer;

// Start server if run directly
if (require.main === module) {
  const server = new DashboardWebSocketServer();
  server.start();
}
