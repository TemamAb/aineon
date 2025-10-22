class DashboardWebSocketManager {
  constructor() {
    this.connection = null;
    this.subscriptions = new Set();
  }

  connect() {
    return new Promise((resolve) => {
      const wsUrl = this.getWebSocketUrl();
      this.connection = new WebSocket(wsUrl);
      this.connection.onopen = () => resolve();
    });
  }

  subscribe(channel) {
    this.subscriptions.add(channel);
  }

  getWebSocketUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/aineon-live`;
  }
}

const websocketManager = new DashboardWebSocketManager();
export default websocketManager;
