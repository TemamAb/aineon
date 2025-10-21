<template>
  <div class="flash-loan-module">
    <div class="module-header">
      <h3>⚡ Flash Loan System</h3>
      <div class="view-toggle" @click="toggleView">
        {{ isAdvancedView ? 'Default' : 'Advanced' }}
      </div>
    </div>
    
    <div v-if="!isAdvancedView" class="default-view">
      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-value">{{ utilizationPercentage }}%</div>
          <div class="metric-label">Utilization</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{{ roiPercentage }}%</div>
          <div class="metric-label">ROI</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{{ providerPerformance }}/10</div>
          <div class="metric-label">Provider Performance</div>
        </div>
      </div>
    </div>

    <div v-else class="advanced-view">
      <div class="advanced-metrics">
        <div class="metric-section">
          <h4>Provider Performance</h4>
          <div class="provider-list">
            <div v-for="provider in providers" :key="provider.name" class="provider-item">
              <span class="provider-name">{{ provider.name }}</span>
              <span class="provider-metric">{{ provider.capacity }}M</span>
              <span class="provider-metric">{{ provider.successRate }}%</span>
            </div>
          </div>
        </div>
        
        <div class="metric-section">
          <h4>Loan Analytics</h4>
          <div class="analytics-grid">
            <div>Total Loans: {{ totalLoans }}</div>
            <div>Avg Duration: {{ avgDuration }}s</div>
            <div>Success Rate: {{ successRate }}%</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'FlashLoanSystem',
  data() {
    return {
      isAdvancedView: false,
      utilizationPercentage: 68,
      roiPercentage: 3.8,
      providerPerformance: 8.7,
      providers: [
        { name: 'Aave V3', capacity: 28.4, successRate: 98.2 },
        { name: 'dYdX', capacity: 22.1, successRate: 99.1 },
        { name: 'Uniswap', capacity: 15.7, successRate: 97.3 }
      ],
      totalLoans: 1247,
      avgDuration: 45,
      successRate: 98.6
    }
  },
  methods: {
    toggleView() {
      this.isAdvancedView = !this.isAdvancedView;
    }
  }
}
</script>

<style scoped>
.flash-loan-module {
  border: 1px solid #e1e8ed;
  border-radius: 8px;
  padding: 16px;
  background: white;
}

.module-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.view-toggle {
  background: #2563eb;
  color: white;
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.metric-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 12px;
}

.metric-card {
  text-align: center;
  padding: 12px;
  background: #f8fafc;
  border-radius: 6px;
}

.metric-value {
  font-size: 20px;
  font-weight: bold;
  color: #2563eb;
}

.metric-label {
  font-size: 12px;
  color: #64748b;
}

.advanced-metrics {
  display: grid;
  gap: 20px;
}

.provider-list {
  display: grid;
  gap: 8px;
}

.provider-item {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  padding: 8px;
  background: #f8fafc;
  border-radius: 4px;
}

.analytics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 12px;
}
</style>
