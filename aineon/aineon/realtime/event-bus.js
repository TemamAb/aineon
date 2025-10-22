// Cross-Module Event System
const EventEmitter = require('events');

class AIEventBus extends EventEmitter {
  constructor() {
    super();
    this.setMaxListeners(50); // Allow multiple module subscriptions
  }

  publishAISignal(signalType, data) {
    this.emit(`ai:${signalType}`, {
      timestamp: Date.now(),
      source: 'ai-orchestrator',
      data,
      confidence: data.confidence || 0.85
    });
  }

  subscribeToAISignals(moduleId, signalTypes, handler) {
    signalTypes.forEach(signalType => {
      this.on(`ai:${signalType}`, (data) => {
        if (this.shouldProcessSignal(moduleId, data)) {
          handler(data);
        }
      });
    });
  }

  shouldProcessSignal(moduleId, signal) {
    // Filter signals based on module requirements and confidence
    return signal.confidence > this.getModuleThreshold(moduleId);
  }

  getModuleThreshold(moduleId) {
    const thresholds = {
      'trading-params': 0.90,
      'execution': 0.85,
      'monitoring': 0.75
    };
    return thresholds[moduleId] || 0.80;
  }
}

module.exports = new AIEventBus();
