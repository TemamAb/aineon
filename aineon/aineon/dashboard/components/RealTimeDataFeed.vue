<template>
  <div class="data-feed">
    <div class="metrics-grid">
      <div class="metric-card" v-for="metric in liveMetrics" :key="metric.id">
        <h4>{{ metric.name }}</h4>
        <div class="metric-value">{{ metric.value }}</div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'RealTimeDataFeed',
  data() {
    return {
      liveMetrics: [
        { id: 'profit', name: 'Total Profit', value: '$2.1M' },
        { id: 'accuracy', name: 'AI Confidence', value: '94.8%' },
        { id: 'speed', name: 'Decision Speed', value: '450ms' }
      ]
    }
  }
}
</script>
