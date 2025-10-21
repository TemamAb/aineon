<template>
  <div class="optimization-module">
    <div class="module-header">
      <h3>í´„ Optimization Engine</h3>
      <div class="view-toggle" @click="toggleView">
        {{ isAdvancedView ? 'Default' : 'Advanced' }}
      </div>
    </div>
    
    <div v-if="!isAdvancedView" class="default-view">
      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-value">{{ activeStrategies }}</div>
          <div class="metric-label">Active Strategies</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{{ researchPipeline }}</div>
          <div class="metric-label">Research Pipeline</div>
        </div>
      </div>
    </div>

    <div v-else class="advanced-view">
      <div class="advanced-metrics">
        <div class="metric-section">
          <h4>Strategy Research Pipeline</h4>
          <div class="pipeline-list">
            <div v-for="item in pipeline" :key="item.id" class="pipeline-item">
              <span class="pipeline-name">{{ item.name }}</span>
              <span class="pipeline-status" :class="item.status">{{ item.status }}</span>
              <span class="pipeline-progress">{{ item.progress }}%</span>
            </div>
          </div>
        </div>
        
        <div class="metric-section">
          <h4>Allocation Weights</h4>
          <div class="allocation-grid">
            <div v-for="strategy in allocations" :key="strategy.name" class="allocation-item">
              <span class="strategy-name">{{ strategy.name }}</span>
              <div class="weight-bar">
                <div class="weight-fill" :style="{ width: strategy.weight + '%' }"></div>
              </div>
              <span class="weight-value">{{ strategy.weight }}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'OptimizationEngine',
  data() {
    return {
      isAdvancedView: false,
      activeStrategies: 15,
      researchPipeline: 8,
      pipeline: [
        { id: 1, name: 'Cross-Chain MEV', status: 'testing', progress: 85 },
        { id: 2, name: 'Omnichain Yield', status: 'research', progress: 45 },
        { id: 3, name: 'Liquidity Arb V2', status: 'deployed', progress: 100 }
      ],
      allocations: [
        { name: 'Flash Loan Arb', weight: 25 },
        { name: 'Cross-Chain', weight: 20 },
        { name: 'CEX/DEX', weight: 15 },
        { name: 'Triangular', weight: 12 },
        { name: 'Market Making', weight: 10 }
      ]
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
.optimization-module {
  border: 1px solid #e1e8ed;
  border-radius: 8px;
  padding: 16px;
  background: white;
}

.pipeline-list {
  display: grid;
  gap: 8px;
}

.pipeline-item {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr;
  gap: 12px;
  padding: 8px;
  background: #f8fafc;
  border-radius: 4px;
}

.pipeline-status.research { color: #f59e0b; }
.pipeline-status.testing { color: #2563eb; }
.pipeline-status.deployed { color: #10b981; }

.allocation-grid {
  display: grid;
  gap: 8px;
}

.allocation-item {
  display: grid;
  grid-template-columns: 1fr 2fr 1fr;
  gap: 12px;
  align-items: center;
}

.weight-bar {
  height: 8px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
}

.weight-fill {
  height: 100%;
  background: #2563eb;
  transition: width 0.3s ease;
}
</style>
