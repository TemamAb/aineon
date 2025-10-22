<template>
  <div class="ai-context-panel">
    <h3>í·  AI Context</h3>
    <div class="context-item">
      <span class="label">Market Regime:</span>
      <span class="value">{{ context.regime }}</span>
    </div>
    <div class="context-item">
      <span class="label">Confidence:</span>
      <span class="value">{{ (context.confidence * 100).toFixed(1) }}%</span>
    </div>
    <div class="recommendations">
      <h4>Recommendations</h4>
      <div class="recommendation" v-for="rec in context.recommendations" :key="rec.id">
        {{ rec.text }}
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'AIContextPanel',
  props: {
    context: Object
  }
}
</script>
