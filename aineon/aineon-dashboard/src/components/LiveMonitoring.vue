<template>
  <div class="live-monitoring">
    <!-- Performance Metrics Row -->
    <div class="dashboard-grid">
      <!-- Performance Card -->
      <div class="metric-card grafana-panel">
        <div class="grafana-panel-header">
          <h3 class="grafana-panel-title">Ì≥à Performance</h3>
          <span class="live-indicator">‚óè LIVE</span>
        </div>
        <div class="metric-content">
          <div class="metric-value large">${{ formatNumber(liveProfit) }}</div>
          <div class="metric-label">Live P&L / Minute</div>
          <div class="metric-trend trend-positive">‚ñ≤ +2.3% Today</div>
        </div>
        <div class="metric-grid">
          <div class="metric-item">
            <div class="metric-value">${{ formatNumber(dailyProfit) }}</div>
            <div class="metric-label">Daily</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">${{ formatNumber(monthlyProfit) }}</div>
            <div class="metric-label">Monthly</div>
          </div>
        </div>
      </div>

      <!-- Trade Analytics Card -->
      <div class="metric-card grafana-panel">
        <div class="grafana-panel-header">
          <h3 class="grafana-panel-title">Ì≥ä Trade Analytics</h3>
        </div>
        <div class="metric-grid">
          <div class="metric-item">
            <div class="metric-value">${{ formatNumber(profitPerTrade) }}</div>
            <div class="metric-label">Profit/Trade</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">{{ tradesPerHour }}</div>
            <div class="metric-label">Trades/Hour</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">{{ tradesPerDay }}</div>
            <div class="metric-label">Trades/Day</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">{{ winRate }}%</div>
            <div class="metric-label">Win Rate</div>
          </div>
        </div>
      </div>

      <!-- AI Intelligence Card -->
      <div class="metric-card grafana-panel">
        <div class="grafana-panel-header">
          <h3 class="grafana-panel-title">Ì∑† AI Intelligence</h3>
        </div>
        <div class="metric-content">
          <div class="metric-value">{{ aiAccuracy }}%</div>
          <div class="metric-label">Decision Accuracy</div>
          <div class="status-item">
            <span class="status-indicator status-online"></span>
            <span>Market Regime: {{ marketRegime }}</span>
          </div>
          <div class="status-item">
            <span class="status-indicator status-online"></span>
            <span>Confidence: {{ aiConfidence }}%</span>
          </div>
        </div>
      </div>

      <!-- Risk Metrics Card -->
      <div class="metric-card grafana-panel">
        <div class="grafana-panel-header">
          <h3 class="grafana-panel-title">Ìª°Ô∏è Risk Metrics</h3>
        </div>
        <div class="metric-grid">
          <div class="metric-item">
            <div class="metric-value">{{ maxDrawdown }}%</div>
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-trend trend-positive">‚ñº -0.2%</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">{{ dailyVar }}%</div>
            <div class="metric-label">Daily VaR</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">{{ sharpeRatio }}</div>
            <div class="metric-label">Sharpe Ratio</div>
          </div>
        </div>
      </div>

      <!-- Execution Engine Card -->
      <div class="metric-card grafana-panel">
        <div class="grafana-panel-header">
          <h3 class="grafana-panel-title">‚ö° Execution Engine</h3>
        </div>
        <div class="metric-grid">
          <div class="metric-item">
            <div class="metric-value">{{ executionSpeed }}ms</div>
            <div class="metric-label">Speed</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">{{ successRate }}%</div>
            <div class="metric-label">Success Rate</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">{{ priceImpact }}%</div>
            <div class="metric-label">Price Impact</div>
          </div>
        </div>
        <div class="status-item">
          <span class="status-indicator status-online"></span>
          <span>Gasless Mode: Active ‚õΩ</span>
        </div>
      </div>

      <!-- Capital Management Card -->
      <div class="metric-card grafana-panel grid-col-2">
        <div class="grafana-panel-header">
          <h3 class="grafana-panel-title">Ì≤∞ Capital Management</h3>
        </div>
        <div class="metric-grid">
          <div class="metric-item">
            <div class="metric-value">${{ formatNumber(capitalUtilized) }}M</div>
            <div class="metric-label">Utilized</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">{{ utilizationRate }}%</div>
            <div class="metric-label">Utilization Rate</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">{{ activeStrategies }}</div>
            <div class="metric-label">Active Strategies</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">{{ activeChains }}</div>
            <div class="metric-label">Active Chains</div>
          </div>
        </div>
        <div class="progress-bar">
          <div class="progress-label">Capital Efficiency</div>
          <div class="progress-track">
            <div class="progress-fill" :style="{ width: capitalEfficiency + '%' }"></div>
          </div>
          <div class="progress-value">{{ capitalEfficiency }}%</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'

export default {
  name: 'LiveMonitoring',
  setup() {
    // Live metrics data
    const liveProfit = ref(1247)
    const dailyProfit = ref(10842)
    const monthlyProfit = ref(324000)
    const profitPerTrade = ref(1247)
    const tradesPerHour = ref(24.3)
    const tradesPerDay = ref(583)
    const winRate = ref(64.7)
    const aiAccuracy = ref(97.8)
    const aiConfidence = ref(94.8)
    const marketRegime = ref('High Volatility')
    const maxDrawdown = ref(6.2)
    const dailyVar = ref(1.8)
    const sharpeRatio = ref(2.1)
    const executionSpeed = ref(48)
    const successRate = ref(99.3)
    const priceImpact = ref(0.08)
    const capitalUtilized = ref(92.4)
    const utilizationRate = ref(92.4)
    const activeStrategies = ref(15)
    const activeChains = ref(8)
    const capitalEfficiency = ref(94.2)

    const formatNumber = (num) => {
      if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M'
      } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K'
      }
      return num.toLocaleString()
    }

    // Simulate live updates
    onMounted(() => {
      setInterval(() => {
        liveProfit.value += Math.random() * 100 - 50
        dailyProfit.value += Math.random() * 500 - 250
      }, 2000)
    })

    return {
      liveProfit,
      dailyProfit,
      monthlyProfit,
      profitPerTrade,
      tradesPerHour,
      tradesPerDay,
      winRate,
      aiAccuracy,
      aiConfidence,
      marketRegime,
      maxDrawdown,
      dailyVar,
      sharpeRatio,
      executionSpeed,
      successRate,
      priceImpact,
      capitalUtilized,
      utilizationRate,
      activeStrategies,
      activeChains,
      capitalEfficiency,
      formatNumber
    }
  }
}
</script>

<style scoped>
.live-monitoring {
  padding: var(--space-4);
}

.metric-content {
  text-align: center;
  margin-bottom: var(--space-4);
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: var(--space-3);
}

.metric-item {
  text-align: center;
  padding: var(--space-3);
  background: rgba(255, 255, 255, 0.05);
  border-radius: var(--border-radius);
}

.status-item {
  display: flex;
  align-items: center;
  margin: var(--space-2) 0;
  font-size: 0.9em;
}

.progress-bar {
  margin-top: var(--space-4);
}

.progress-label {
  font-size: 0.9em;
  color: var(--grafana-text-secondary);
  margin-bottom: var(--space-2);
}

.progress-track {
  background: var(--grafana-dark);
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--grafana-blue), var(--grafana-green));
  transition: width 0.3s ease;
}

.progress-value {
  text-align: right;
  font-size: 0.8em;
  color: var(--grafana-text-secondary);
  margin-top: var(--space-1);
}

.live-indicator {
  color: var(--grafana-success);
  font-size: 0.8em;
  animation: pulse 2s infinite;
}
</style>
