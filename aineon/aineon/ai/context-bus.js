class AIContextBus {
  constructor() {
    this.context = new Map();
    this.subscribers = new Map();
  }

  updateContext(moduleId, data, confidence) {
    const timestamp = Date.now();
    const contextUpdate = {
      moduleId,
      data,
      confidence,
      timestamp
    };

    this.context.set(moduleId, contextUpdate);
    const consensus = this.calculateConsensus();
    this.context.set('consensus', consensus);
    this.notifySubscribers(moduleId, contextUpdate, consensus);
  }

  getUnifiedIntelligence() {
    const modules = ['strategicAI', 'tacticalAI', 'riskAssessment', 'patternRecognition'];
    const intelligence = {};

    modules.forEach(moduleId => {
      const ctx = this.context.get(moduleId);
      if (ctx && ctx.confidence > 0.75) {
        intelligence[moduleId] = ctx.data;
      }
    });

    return {
      ...intelligence,
      consensus: this.context.get('consensus'),
      timestamp: Date.now()
    };
  }

  calculateConsensus() {
    const modules = Array.from(this.context.values())
      .filter(ctx => ctx.moduleId !== 'consensus');

    if (modules.length === 0) return { score: 0, agreement: 'low' };

    const avgConfidence = modules.reduce((sum, m) => sum + m.confidence, 0) / modules.length;
    return {
      score: avgConfidence,
      agreement: avgConfidence > 0.8 ? 'high' : avgConfidence > 0.6 ? 'medium' : 'low',
      participatingModules: modules.length
    };
  }

  subscribe(moduleId, callback) {
    if (!this.subscribers.has(moduleId)) {
      this.subscribers.set(moduleId, new Set());
    }
    this.subscribers.get(moduleId).add(callback);
  }

  notifySubscribers(sourceModule, update, consensus) {
    this.subscribers.forEach((callbacks, moduleId) => {
      if (moduleId !== sourceModule) {
        callbacks.forEach(callback => callback(update, consensus));
      }
    });
  }
}

module.exports = new AIContextBus();
