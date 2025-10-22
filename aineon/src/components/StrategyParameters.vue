<template>
  <div class="strategy-parameters">
    <!-- AI Recommendation Banner -->
    <div class="ai-recommendation-banner" v-if="showAIRecommendation && !userHasOverridden">
      <div class="banner-content">
        <h3>Ì∑† AI Recommended Strategy</h3>
        <p>Based on current market analysis with <strong>{{ aiConfidence }}% confidence</strong></p>
        <div class="banner-actions">
          <button @click="acceptAIStrategy" class="btn-primary">Use AI Strategy</button>
          <button @click="startCustomSetup" class="btn-secondary">Custom Setup</button>
        </div>
      </div>
    </div>

    <!-- Main Dashboard -->
    <div class="dashboard">
      <div class="control-header" v-if="userHasOverridden">
        <h3>ÌæõÔ∏è Custom Strategy Setup</h3>
        <button @click="resetToAIRecommended" class="btn-secondary">Reset to AI Recommended</button>
      </div>

      <!-- Strategy Cards -->
      <div class="strategy-card">
        <h3>Ì≤∞ Profit Targets</h3>
        <div class="parameter-grid">
          <div class="parameter-item" v-for="(target, key) in profitTargets" :key="key">
            <label>{{ target.label }}</label>
            <input 
              v-model="target.value" 
              type="text" 
              :placeholder="`AI: ${target.aiValue}`"
              @input="onUserInput"
            >
          </div>
        </div>
      </div>

      <div class="strategy-card">
        <h3>Ìª°Ô∏è Risk Parameters</h3>
        <div class="parameter-grid">
          <div class="parameter-item" v-for="(param, key) in riskParameters" :key="key">
            <label>{{ param.label }}</label>
            <input 
              v-model="param.value" 
              type="text" 
              :placeholder="`AI: ${param.aiValue}%`"
              @input="onUserInput"
            >
          </div>
        </div>
      </div>

      <!-- Deploy Button -->
      <div class="deploy-section">
        <button @click="deployStrategy" class="btn-primary deploy-btn">
          Ì∫Ä {{ userHasOverridden ? 'Deploy Custom Strategy' : 'Deploy AI Strategy' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'

export default {
  name: 'StrategyParameters',
  setup() {
    const showAIRecommendation = ref(true)
    const userHasOverridden = ref(false)
    const aiConfidence = ref(94.8)

    const profitTargets = ref({
      daily: { value: '', aiValue: '29,928', label: 'Daily Target' },
      weekly: { value: '', aiValue: '209,496', label: 'Weekly Target' },
      monthly: { value: '', aiValue: '897,840', label: 'Monthly Target' }
    })

    const riskParameters = ref({
      maxDrawdown: { value: '', aiValue: '6.2', label: 'Max Drawdown' },
      dailyVaR: { value: '', aiValue: '1.8', label: 'Daily VaR' }
    })

    const onUserInput = () => {
      userHasOverridden.value = true
    }

    const acceptAIStrategy = () => {
      Object.keys(profitTargets.value).forEach(key => {
        profitTargets.value[key].value = profitTargets.value[key].aiValue
      })
      Object.keys(riskParameters.value).forEach(key => {
        riskParameters.value[key].value = riskParameters.value[key].aiValue
      })
      userHasOverridden.value = false
      showAIRecommendation.value = false
    }

    const startCustomSetup = () => {
      userHasOverridden.value = true
      showAIRecommendation.value = false
    }

    const resetToAIRecommended = () => {
      acceptAIStrategy()
    }

    const deployStrategy = () => {
      alert(userHasOverridden.value ? 'Ì∫Ä Custom Strategy Deployed!' : 'Ì∫Ä AI Strategy Deployed!')
    }

    return {
      showAIRecommendation,
      userHasOverridden,
      aiConfidence,
      profitTargets,
      riskParameters,
      onUserInput,
      acceptAIStrategy,
      startCustomSetup,
      resetToAIRecommended,
      deployStrategy
    }
  }
}
</script>

<style scoped>
.strategy-parameters {
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
}

.ai-recommendation-banner {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px;
  border-radius: 10px;
  margin-bottom: 20px;
  text-align: center;
}

.control-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #2a2a2a;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.strategy-card {
  background: #1a1a1a;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
  border: 1px solid #333;
}

.parameter-grid {
  display: grid;
  gap: 15px;
}

.parameter-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.parameter-item label {
  font-weight: bold;
  color: #ccc;
}

.parameter-item input {
  padding: 10px;
  border: 1px solid #444;
  border-radius: 4px;
  background: #2a2a2a;
  color: white;
}

.btn-primary {
  background: #667eea;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 16px;
}

.btn-secondary {
  background: #555;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
}

.deploy-section {
  text-align: center;
  margin-top: 30px;
}

.deploy-btn {
  font-size: 18px;
  padding: 15px 30px;
}
</style>
