<template>
  <div class="optimization-panel">
    <h3>⚡ Optimization</h3>
    <div class="stats-grid">
      <div class="stat">
        <span class="label">Cycles:</span>
        <span class="value">{{ stats.cycleCount }}</span>
      </div>
      <div class="stat">
        <span class="label">Multiplier:</span>
        <span class="value">{{ stats.compoundingMultiplier.toFixed(3) }}x</span>
      </div>
      <div class="stat">
        <span class="label">Efficiency:</span>
        <span class="value">{{ stats.efficiencyScore.toFixed(1) }}%</span>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'OptimizationPanel',
  props: {
    stats: Object
  }
}
</script>
