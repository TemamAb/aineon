import { ref } from 'vue';

export interface WebSocketMessage {
  type: string;
  payload: any;
  timestamp: number;
}

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private isConnected = ref(false);
  private messageHandlers: Map<string, Function[]> = new Map();

  connect(url: string = 'ws://localhost:5001') { // Port 5001
    try {
      this.ws = new WebSocket(url);
      
      this.ws.onopen = () => {
        console.log('í´Œ Master Dashboard WebSocket connected');
        this.isConnected.value = true;
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      };

      this.ws.onclose = () => {
        console.log('í´Œ Master Dashboard WebSocket disconnected');
        this.isConnected.value = false;
        this.handleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('Master Dashboard WebSocket error:', error);
      };

    } catch (error) {
      console.error('Failed to connect to Master Dashboard WebSocket:', error);
    }
  }

  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`í´„ Reconnecting... Attempt ${this.reconnectAttempts}`);
        this.connect();
      }, 3000);
    }
  }

  private handleMessage(message: WebSocketMessage) {
    const handlers = this.messageHandlers.get(message.type) || [];
    handlers.forEach(handler => handler(message.payload));
  }

  on(messageType: string, handler: Function) {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, []);
    }
    this.messageHandlers.get(messageType)!.push(handler);
  }

  getConnectionStatus() {
    return this.isConnected;
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export const webSocketService = new WebSocketService();
