const WebSocket = require('ws');

class WebSocketManager {
  constructor() {
    this.connections = new Map();
    this.subscriptions = new Map();
  }

  initializeServer(server) {
    this.wss = new WebSocket.Server({ server, path: '/aineon-live' });

    this.wss.on('connection', (ws, request) => {
      const clientId = this.generateClientId();
      this.connections.set(clientId, ws);

      ws.on('message', (message) => {
        this.handleMessage(clientId, message);
      });

      ws.on('close', () => {
        this.connections.delete(clientId);
      });

      this.sendToClient(clientId, {
        type: 'connection_established',
        clientId,
        timestamp: Date.now()
      });
    });

    this.startMetricsBroadcast();
  }

  handleMessage(clientId, message) {
    try {
      const data = JSON.parse(message);
      
      switch (data.type) {
        case 'subscribe':
          this.handleSubscription(clientId, data.channels);
          break;
        case 'unsubscribe':
          this.handleUnsubscription(clientId, data.channels);
          break;
      }
    } catch (error) {
      console.error('Error handling message:', error);
    }
  }

  handleSubscription(clientId, channels) {
    if (!this.subscriptions.has(clientId)) {
      this.subscriptions.set(clientId, new Set());
    }
    channels.forEach(channel => {
      this.subscriptions.get(clientId).add(channel);
    });
  }

  broadcastMetrics(metrics) {
    const message = JSON.stringify({
      type: 'live_metrics',
      data: metrics,
      timestamp: Date.now()
    });

    this.connections.forEach((ws) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(message);
      }
    });
  }

  startMetricsBroadcast() {
    setInterval(() => {
      this.broadcastMetrics({
        decisionSpeed: 450,
        aiConfidence: 94.8,
        simulationAccuracy: 97.3,
        gasEfficiency: 92,
        totalProfit: 2100000,
        activeStrategies: 8,
        dailyTransactions: 1847
      });
    }, 1000);
  }

  generateClientId() {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  sendToClient(clientId, data) {
    const ws = this.connections.get(clientId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data));
    }
  }
}

module.exports = new WebSocketManager();
