<template>
  <div class="simulator">
    <div class="simulation-controls">
      <button @click="runBacktest" :disabled="isSimulating">
        {{ isSimulating ? 'Running...' : 'Run Backtest' }}
      </button>
      <div class="accuracy-display">
        Simulation Accuracy: {{ accuracy }}%
      </div>
    </div>
    <div class="results-grid">
      <div class="chart-container">
        <!-- Performance chart would go here -->
      </div>
      <div class="metrics-panel">
        <div v-for="metric in simulationMetrics" :key="metric.name" class="metric">
          <span class="metric-name">{{ metric.name }}:</span>
          <span class="metric-value">{{ metric.value }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'StrategySimulator',
  data() {
    return {
      isSimulating: false,
      accuracy: 97.3,
      simulationMetrics: [
        { name: 'Sharpe Ratio', value: '2.1' },
        { name: 'Max Drawdown', value: '-8.2%' },
        { name: 'Win Rate', value: '64.7%' },
        { name: 'Profit Factor', value: '1.8' }
      ]
    }
  },
  methods: {
    async runBacktest() {
      this.isSimulating = true;
      // Connect to genetic-optimizer.js
      await this.$store.dispatch('ai/runOptimization');
      this.isSimulating = false;
    }
  }
}
</script>
