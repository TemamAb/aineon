// Real-time Trading State Management
import { defineStore } from 'pinia';

export const useTradingStore = defineStore('trading', {
  state: () => ({
    currentStrategy: null,
    livePositions: [],
    pendingOrders: [],
    performanceMetrics: {
      totalProfit: 2100000,
      dailyTrades: 1847,
      winRate: 0.647,
      sharpeRatio: 2.1
    },
    aiParameters: {
      riskTolerance: 0.02,
      positionSize: 0.05,
      maxDrawdown: 0.08
    }
  }),

  actions: {
    async deployOptimizedStrategy(optimizationResults) {
      // Connect to deployment module
      this.currentStrategy = optimizationResults;
      await this.$api.deployStrategy(optimizationResults);
    },

    updateLiveMetrics(metrics) {
      this.performanceMetrics = { ...this.performanceMetrics, ...metrics };
    }
  }
});
