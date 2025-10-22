<template>
  <div class="flash-loan-module">
    <div class="module-header">
      <h3>⚡ Flash Loan System</h3>
      <div class="status-indicator" :class="systemStatus">
        {{ systemStatus.toUpperCase() }}
      </div>
    </div>
    
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value">{{ utilizationRate }}%</div>
        <div class="metric-label">Utilization Rate</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{{ successRate }}%</div>
        <div class="metric-label">Success Rate</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${{ totalVolume }}M</div>
        <div class="metric-label">Total Volume</div>
      </div>
    </div>

    <div class="providers-list">
      <h4>Active Providers</h4>
      <div v-for="provider in providers" :key="provider.name" class="provider-item">
        <span class="provider-name">{{ provider.name }}</span>
        <span class="provider-metric">{{ provider.utilization }}% util</span>
        <span class="provider-metric">{{ provider.rate }}% APR</span>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'FlashLoanSystem',
  data() {
    return {
      systemStatus: 'active',
      utilizationRate: 68,
      successRate: 98.7,
      totalVolume: 142.3,
      providers: [
        { name: 'Aave V3', utilization: 72, rate: 3.2 },
        { name: 'dYdX', utilization: 65, rate: 2.8 },
        { name: 'Uniswap V3', utilization: 58, rate: 3.8 }
      ]
    }
  }
}
</script>
