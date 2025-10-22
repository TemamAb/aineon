<template>
  <div class="strategy-parameters">
    <!-- AI Recommendation Banner (Dismissible) -->
    <div class="ai-recommendation-banner grafana-panel" v-if="showAIRecommendation && !userHasOverridden">
      <div class="banner-content">
        <div class="banner-icon">Ì∑†</div>
        <div class="banner-text">
          <h3>AI Recommended Strategy</h3>
          <p>Based on current market analysis with <strong>{{ aiConfidence }}% confidence</strong></p>
          <div class="banner-details">
            <span class="detail-item">ÌæØ Daily Target: ${{ getAIValue('daily') }}</span>
            <span class="detail-item">Ìª°Ô∏è Max Drawdown: {{ getAIValue('maxDrawdown') }}%</span>
            <span class="detail-item">‚ö° Gasless: Enabled</span>
          </div>
        </div>
        <div class="banner-actions">
          <button @click="acceptAIStrategy" class="grafana-btn-primary">Use AI Strategy</button>
          <button @click="startCustomSetup" class="grafana-btn">Custom Setup</button>
          <button @click="dismissBanner" class="grafana-btn-text">Dismiss</button>
        </div>
      </div>
    </div>

    <!-- User Control Header -->
    <div class="control-header grafana-panel" v-if="userHasOverridden">
      <div class="control-info">
        <span class="control-icon">ÌæõÔ∏è</span>
        <div class="control-text">
          <h3>Custom Strategy Setup</h3>
          <p>You're in full control. AI suggestions shown as reference.</p>
        </div>
      </div>
      <div class="control-actions">
        <button @click="resetToAIRecommended" class="grafana-btn">
          Reset to AI Recommended
        </button>
        <button @click="showAIRecommendation = true" class="grafana-btn-text">
          Show AI Suggestions
        </button>
      </div>
    </div>

    <div class="dashboard-grid">
      <!-- Strategy Control Center -->
      <div class="metric-card grafana-panel grid-col-2">
        <div class="grafana-panel-header">
          <h3 class="grafana-panel-title">
            {{ userHasOverridden ? 'ÌæõÔ∏è Custom Strategy' : 'Ì∑† AI Recommended Strategy' }}
          </h3>
          <div class="strategy-controls">
            <div class="control-badge" :class="userHasOverridden ? 'custom' : 'ai'">
              {{ userHasOverridden ? 'CUSTOM' : 'AI OPTIMIZED' }}
            </div>
            <div class="confidence-display" v-if="!userHasOverridden">
              {{ aiConfidence }}% Confidence
            </div>
          </div>
        </div>
        
        <div class="strategy-metrics">
          <div class="metric-item">
            <div class="metric-value">{{ marketRegime }}</div>
            <div class="metric-label">Market Regime</div>
          </div>
          <div class="metric-item">
            <div class="metric-value">{{ estimatedCapacity }}M</div>
            <div class="metric-label">Market Capacity</div>
          </div>
          <div class="metric-item">
            <div class="metric-value" :class="userHasOverridden ? 'custom-value' : 'ai-value'">
              {{ userHasOverridden ? 'Custom' : 'AI Optimized' }}
            </div>
            <div class="metric-label">Strategy Type</div>
          </div>
        </div>

        <div class="deployment-controls">
          <button @click="deployStrategy" class="grafana-btn-primary" :class="{ 'custom-deploy': userHasOverridden }">
            Ì∫Ä {{ userHasOverridden ? 'Deploy Custom Strategy' : 'Deploy AI Strategy' }}
          </button>
          <button @click="generateAIStrategy" class="grafana-btn" v-if="userHasOverridden">
            Ì¥ñ Get AI Recommendation
          </button>
          <button @click="clearAllInputs" class="grafana-btn-text" v-if="userHasOverridden">
            Clear All
          </button>
        </div>
      </div>

      <!-- Profit Targets Card -->
      <div class="metric-card grafana-panel">
        <div class="grafana-panel-header">
          <h3 class="grafana-panel-title">Ì≤∞ Profit Targets</h3>
          <div class="parameter-control-mode">
            <span class="mode-indicator" :class="userHasOverridden ? 'custom-mode' : 'ai-mode'">
              {{ userHasOverridden ? 'Manual' : 'AI' }}
            </span>
          </div>
        </div>
        <div class="parameter-grid">
          <div class="parameter-item" v-for="(target, key) in profitTargets" :key="key">
            <label class="parameter-label">{{ target.label }}</label>
            <div class="parameter-control">
              <input 
                v-model="target.value" 
                type="text" 
                class="grafana-input"
                :placeholder="`AI: ${target.aiValue}`"
                @input="onUserInput"
                @focus="onParameterFocus(key)"
              >
              <select v-model="target.currency" class="grafana-select" @change="onUserInput">
                <option value="USD">USD</option>
                <option value="ETH">ETH</option>
              </select>
            </div>
            <div class="parameter-suggestion" v-if="!target.value">
              AI suggests: {{ target.aiValue }} {{ target.currency }}
            </div>
            <div class="parameter-user" v-else>
              Your value
            </div>
          </div>
        </div>
        <div class="parameter-actions">
          <button @click="fillAIProfitTargets" class="grafana-btn-text">
            Fill AI Recommendations
          </button>
        </div>
      </div>

      <!-- Risk Parameters Card -->
      <div class="metric-card grafana-panel">
        <div class="grafana-panel-header">
          <h3 class="grafana-panel-title">Ìª°Ô∏è Risk Parameters</h3>
          <div class="risk-controls">
            <button @click="setRiskProfile('conservative')" class="risk-btn" :class="{ active: riskProfile === 'conservative' }">
              Low Risk
            </button>
            <button @click="setRiskProfile('moderate')" class="risk-btn" :class="{ active: riskProfile === 'moderate' }">
              Moderate
            </button>
            <button @click="setRiskProfile('aggressive')" class="risk-btn" :class="{ active: riskProfile === 'aggressive' }">
              High Risk
            </button>
          </div>
        </div>
        <div class="parameter-grid">
          <div class="parameter-item" v-for="(param, key) in riskParameters" :key="key">
            <label class="parameter-label">{{ param.label }}</label>
            <div class="parameter-control">
              <input 
                v-model="param.value" 
                type="text" 
                class="grafana-input"
                :placeholder="`AI: ${param.aiValue}%`"
                @input="onUserInput"
              >
              <span class="parameter-unit">%</span>
            </div>
            <div class="parameter-suggestion" v-if="!param.value">
              AI: {{ param.aiValue }}%
            </div>
            <div class="parameter-user" v-else>
              Your value
            </div>
          </div>
        </div>
        <div class="parameter-actions">
          <button @click="fillAIRiskParameters" class="grafana-btn-text">
            Use AI Risk Settings
          </button>
        </div>
      </div>

      <!-- Strategy Allocation Card -->
      <div class="metric-card grafana-panel grid-col-2">
        <div class="grafana-panel-header">
          <h3 class="grafana-panel-title">Ì≥à Strategy Allocation</h3>
          <div class="allocation-presets">
            <span class="preset-label">Presets:</span>
            <button @click="setAllocationPreset('volatility')" class="preset-btn">High Volatility</button>
            <button @click="setAllocationPreset('trend')" class="preset-btn">Trend Following</button>
            <button @click="setAllocationPreset('balanced')" class="preset-btn">Balanced</button>
          </div>
        </div>
        <div class="allocation-grid">
          <div class="allocation-item" v-for="(strategy, index) in strategyAllocation" :key="strategy.name">
            <div class="allocation-info">
              <span class="strategy-name">{{ strategy.name }}</span>
              <span class="strategy-type">{{ strategy.type }}</span>
            </div>
            <div class="allocation-control">
              <input 
                v-model="strategy.allocation" 
                type="range" 
                min="0" 
                max="100"
                class="allocation-slider"
                @input="onAllocationChange"
              >
              <input 
                v-model="strategy.allocation" 
                type="number" 
                class="allocation-input"
                min="0"
                max="100"
                @input="onAllocationInput(index)"
              >
              <span class="allocation-unit">%</span>
            </div>
            <div class="allocation-suggestion">
              <span class="suggestion-label" v-if="!strategy.allocation">AI: {{ strategy.aiAllocation }}%</span>
              <span class="user-label" v-else>Your allocation</span>
              <button 
                @click="setStrategyToAI(index)" 
                class="suggestion-btn"
                :class="{ visible: strategy.allocation && parseInt(strategy.allocation) !== strategy.aiAllocation }"
              >
                Use AI
              </button>
            </div>
          </div>
        </div>
        <div class="allocation-summary">
          <div class="total-allocation" :class="{ valid: totalAllocation === 100, invalid: totalAllocation !== 100 }">
            Total: {{ totalAllocation }}% {{ totalAllocation !== 100 ? ' (AI will auto-balance)' : '' }}
          </div>
          <div class="allocation-actions">
            <button @click="autoBalanceAllocation" class="grafana-btn-text">
              Auto-balance to 100%
            </button>
            <button @click="fillAIAllocation" class="grafana-btn-text">
              Use AI Allocation
            </button>
          </div>
        </div>
      </div>

      <!-- Execution Settings Card -->
      <div class="metric-card grafana-panel">
        <div class="grafana-panel-header">
          <h3 class="grafana-panel-title">‚ö° Execution Settings</h3>
          <div class="settings-mode">
            <span class="mode-indicator" :class="userHasOverridden ? 'custom-mode' : 'ai-mode'">
              {{ userHasOverridden ? 'Manual' : 'AI Recommended' }}
            </span>
          </div>
        </div>
        <div class="settings-grid">
          <div class="setting-item">
            <label class="setting-label">
              Gasless Mode
              <span class="setting-recommendation">(AI recommends: Enabled)</span>
            </label>
            <div class="setting-control">
              <label class="toggle-switch">
                <input type="checkbox" v-model="gaslessMode" @change="onUserInput">
                <span class="toggle-slider"></span>
              </label>
              <span class="setting-status">{{ gaslessMode ? 'Enabled' : 'Disabled' }}</span>
            </div>
          </div>
          <div class="setting-item">
            <label class="setting-label">
              Price Impact Limit
              <span class="setting-recommendation">(AI recommends: 0.08%)</span>
            </label>
            <div class="setting-control">
              <input 
                v-model="priceImpactLimit" 
                type="text" 
                class="grafana-input"
                placeholder="0.08"
                @input="onUserInput"
              >
              <span class="setting-unit">%</span>
            </div>
          </div>
          <div class="setting-item">
            <label class="setting-label">
              Auto-Optimization
              <span class="setting-recommendation">(AI recommends: Enabled)</span>
            </label>
            <div class="setting-control">
              <label class="toggle-switch">
                <input type="checkbox" v-model="autoOptimization" @change="onUserInput">
                <span class="toggle-slider"></span>
              </label>
              <span class="setting-status">{{ autoOptimization ? 'Enabled' : 'Disabled' }}</span>
            </div>
          </div>
        </div>
        <div class="settings-actions">
          <button @click="resetExecutionToAI" class="grafana-btn-text">
            Reset to AI Recommended
          </button>
        </div>
      </div>
    </div>

    <!-- Deployment Progress Modal -->
    <div v-if="showDeploymentProgress" class="progress-modal">
      <div class="progress-content grafana-panel">
        <h3>
          {{ userHasOverridden ? 'Ì∫Ä Deploying Custom Strategy' : 'Ì∫Ä Deploying AI Optimized Strategy' }}
        </h3>
        <div class="deployment-type" :class="userHasOverridden ? 'custom-deployment' : 'ai-deployment'">
          {{ userHasOverridden ? 'Custom Strategy' : 'AI Optimized Strategy' }}
        </div>
        <div class="progress-steps">
          <div v-for="(step, index) in deploymentSteps" 
               :key="step.id"
               :class="['progress-step', { active: currentStep >= index, completed: currentStep > index }]">
            <div class="step-icon">{{ step.icon }}</div>
            <div class="step-info">
              <div class="step-title">{{ step.title }}</div>
              <div class="step-description">{{ step.description }}</div>
            </div>
            <div class="step-status">
              <span v-if="currentStep > index">‚úÖ</span>
              <span v-else-if="currentStep === index">Ì¥Ñ</span>
              <span v-else>‚è±Ô∏è</span>
            </div>
          </div>
        </div>
        <div class="progress-actions">
          <button @click="cancelDeployment" class="grafana-btn">Cancel</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'

export default {
  name: 'StrategyParameters',
  setup() {
    // AI Strategy State
    const showAIRecommendation = ref(true)
    const userHasOverridden = ref(false)
    const aiConfidence = ref(94.8)
    const marketRegime = ref('High Volatility')
    const estimatedCapacity = ref(142)
    const riskProfile = ref('moderate')

    // User Parameters (initially empty for AI defaults)
    const profitTargets = ref({
      hourly: { value: '', currency: 'USD', aiValue: '1,247', label: 'Hourly Target' },
      daily: { value: '', currency: 'USD', aiValue: '29,928', label: 'Daily Target' },
      weekly: { value: '', currency: 'USD', aiValue: '209,496', label: 'Weekly Target' },
      monthly: { value: '', currency: 'USD', aiValue: '897,840', label: 'Monthly Target' }
    })

    const riskParameters = ref({
      maxDrawdown: { value: '', aiValue: '6.2', label: 'Max Drawdown' },
      dailyVaR: { value: '', aiValue: '1.8', label: 'Daily VaR' },
      positionSize: { value: '', aiValue: '1.5', label: 'Position Size' },
      growthRate: { value: '', aiValue: '0.23', label: 'Growth Rate' }
    })

    const strategyAllocation = ref([
      { name: 'Volatility Arbitrage', type: 'Arbitrage', allocation: '', aiAllocation: 35 },
      { name: 'Mean Reversion', type: 'Statistical', allocation: '', aiAllocation: 25 },
      { name: 'Liquidity Providing', type: 'Yield', allocation: '', aiAllocation: 20 },
      { name: 'Cross-chain Arbitrage', type: 'Arbitrage', allocation: '', aiAllocation: 20 }
    ])

    const gaslessMode = ref(true)
    const priceImpactLimit = ref('')
    const autoOptimization = ref(true)

    // Deployment State
    const showDeploymentProgress = ref(false)
    const currentStep = ref(0)
    const deploymentSteps = ref([
      { id: 'analysis', icon: 'Ì∑†', title: 'Market Analysis', description: 'Analyzing current market conditions' },
      { id: 'validation', icon: 'Ì≥ä', title: 'Strategy Validation', description: 'Running historical backtests' },
      { id: 'allocation', icon: 'Ì≤∞', title: 'Capital Allocation', description: 'Optimizing capital distribution' },
      { id: 'deployment', icon: 'Ì∫Ä', title: 'Live Deployment', description: 'Deploying across 8 chains' },
      { id: 'monitoring', icon: 'Ì±ÅÔ∏è', title: 'Monitoring Setup', description: 'Initializing real-time tracking' }
    ])

    // Computed Properties
    const totalAllocation = computed(() => {
      return strategyAllocation.value.reduce((sum, s) => sum + (parseInt(s.allocation) || 0), 0)
    })

    // Methods
    const onUserInput = () => {
      userHasOverridden.value = true
    }

    const onParameterFocus = (paramKey) => {
      // When user focuses on empty field, show AI value as placeholder
      if (!profitTargets.value[paramKey].value) {
        // The placeholder is already set in the template
      }
    }

    const startCustomSetup = () => {
      userHasOverridden.value = true
      showAIRecommendation.value = false
    }

    const dismissBanner = () => {
      showAIRecommendation.value = false
    }

    const acceptAIStrategy = () => {
      fillAIProfitTargets()
      fillAIRiskParameters()
      fillAIAllocation()
      resetExecutionToAI()
      userHasOverridden.value = false
      showAIRecommendation.value = false
    }

    const resetToAIRecommended = () => {
      if (confirm('Reset all values to AI recommendations?')) {
        acceptAIStrategy()
      }
    }

    const generateAIStrategy = () => {
      showAIRecommendation.value = true
    }

    const fillAIProfitTargets = () => {
      Object.keys(profitTargets.value).forEach(key => {
        profitTargets.value[key].value = profitTargets.value[key].aiValue
      })
      onUserInput()
    }

    const fillAIRiskParameters = () => {
      Object.keys(riskParameters.value).forEach(key => {
        riskParameters.value[key].value = riskParameters.value[key].aiValue
      })
      onUserInput()
    }

    const fillAIAllocation = () => {
      strategyAllocation.value.forEach(strategy => {
        strategy.allocation = strategy.aiAllocation.toString()
      })
      onUserInput()
    }

    const setStrategyToAI = (index) => {
      strategyAllocation.value[index].allocation = strategyAllocation.value[index].aiAllocation.toString()
      onUserInput()
    }

    const resetExecutionToAI = () => {
      gaslessMode.value = true
      priceImpactLimit.value = '0.08'
      autoOptimization.value = true
      onUserInput()
    }

    const clearAllInputs = () => {
      if (confirm('Clear all custom inputs?')) {
        Object.keys(profitTargets.value).forEach(key => {
          profitTargets.value[key].value = ''
        })
        Object.keys(riskParameters.value).forEach(key => {
          riskParameters.value[key].value = ''
        })
        strategyAllocation.value.forEach(strategy => {
          strategy.allocation = ''
        })
        priceImpactLimit.value = ''
        userHasOverridden.value = false
        showAIRecommendation.value = true
      }
    }

    const setRiskProfile = (profile) => {
      riskProfile.value = profile
      const profiles = {
        conservative: { maxDrawdown: '4.0', dailyVaR: '1.2', positionSize: '1.0', growthRate: '0.15' },
        moderate: { maxDrawdown: '6.2', dailyVaR: '1.8', positionSize: '1.5', growthRate: '0.23' },
        aggressive: { maxDrawdown: '8.0', dailyVaR: '2.5', positionSize: '2.0', growthRate: '0.35' }
      }
      
      const settings = profiles[profile]
      Object.keys(settings).forEach(key => {
        if (riskParameters.value[key]) {
          riskParameters.value[key].value = settings[key]
        }
      })
      onUserInput()
    }

    const setAllocationPreset = (preset) => {
      const presets = {
        volatility: [35, 25, 20, 20],
        trend: [40, 20, 25, 15],
        balanced: [25, 25, 25, 25]
      }
      
      const allocation = presets[preset]
      strategyAllocation.value.forEach((strategy, index) => {
        strategy.allocation = allocation[index].toString()
      })
      onUserInput()
    }

    const autoBalanceAllocation = () => {
      if (totalAllocation.value !== 100) {
        const difference = 100 - totalAllocation.value
        const strategiesWithAllocation = strategyAllocation.value.filter(s => s.allocation)
        if (strategiesWithAllocation.length > 0) {
          const adjustment = difference / strategiesWithAllocation.length
          strategiesWithAllocation.forEach(strategy => {
            strategy.allocation = Math.max(0, (parseInt(strategy.allocation) + adjustment)).toFixed(1)
          })
        }
      }
      onUserInput()
    }

    const onAllocationChange = () => {
      onUserInput()
    }

    const onAllocationInput = (index) => {
      // Ensure allocation is within bounds
      const value = parseInt(strategyAllocation.value[index].allocation)
      if (value < 0) strategyAllocation.value[index].allocation = '0'
      if (value > 100) strategyAllocation.value[index].allocation = '100'
      onUserInput()
    }

    const getAIValue = (param) => {
      if (param === 'daily') return profitTargets.value.daily.aiValue
      if (param === 'maxDrawdown') return riskParameters.value.maxDrawdown.aiValue
      return ''
    }

    const deployStrategy = async () => {
      showDeploymentProgress.value = true
      currentStep.value = 0

      // Simulate deployment steps
      for (let i = 0; i < deploymentSteps.value.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 2000))
        currentStep.value = i + 1
      }

      // Completion
      setTimeout(() => {
        showDeploymentProgress.value = false
        currentStep.value = 0
        // Navigate to Live Monitoring
        window.location.hash = '#/live-monitoring'
      }, 2000)
    }

    const cancelDeployment = () => {
      showDeploymentProgress.value = false
      currentStep.value = 0
    }

    // Auto-generate AI strategy on component mount if no user input
    onMounted(() => {
      setTimeout(() => {
        const hasInput = Object.values(profitTargets.value).some(t => t.value !== '')
        if (!hasInput) {
          // Show AI recommendation but don't auto-fill
          showAIRecommendation.value = true
        }
      }, 1000)
    })

    return {
      showAIRecommendation,
      userHasOverridden,
      aiConfidence,
      marketRegime,
      estimatedCapacity,
      riskProfile,
      profitTargets,
      riskParameters,
      strategyAllocation,
      gaslessMode,
      priceImpactLimit,
      autoOptimization,
      showDeploymentProgress,
      currentStep,
      deploymentSteps,
      totalAllocation,
      onUserInput,
      onParameterFocus,
      startCustomSetup,
      dismissBanner,
      acceptAIStrategy,
      resetToAIRecommended,
      generateAIStrategy,
      fillAIProfitTargets,
      fillAIRiskParameters,
      fillAIAllocation,
      setStrategyToAI,
      resetExecutionToAI,
      clearAllInputs,
      setRiskProfile,
      setAllocationPreset,
      autoBalanceAllocation,
      onAllocationChange,
      onAllocationInput,
      getAIValue,
      deployStrategy,
      cancelDeployment
    }
  }
}
</script>

<style scoped>
/* Enhanced styles for user override system */
.control-header {
  background: linear-gradient(135deg, rgba(237, 137, 54, 0.1), rgba(245, 101, 101, 0.1));
  border: 1px solid var(--grafana-warning);
  margin-bottom: var(--space-4);
}

.control-info {
  display: flex;
  align-items: center;
  gap: var(--space-4);
}

.control-icon {
  font-size: 2em;
}

.control-text h3 {
  color: var(--grafana-warning);
  margin-bottom: var(--space-1);
}

.control-actions {
  display: flex;
  gap: var(--space-3);
}

.control-badge {
  padding: 4px 8px;
  border-radius: var(--border-radius-sm);
  font-size: 0.8em;
  font-weight: 600;
}

.control-badge.ai {
  background: rgba(50, 208, 255, 0.2);
  color: var(--grafana-text-accent);
}

.control-badge.custom {
  background: rgba(237, 137, 54, 0.2);
  color: var(--grafana-warning);
}

.confidence-display {
  font-size: 0.9em;
  color: var(--grafana-text-accent);
}

.custom-deploy {
  background: var(--grafana-warning);
  color: var(--grafana-dark);
}

.custom-deploy:hover {
  background: #e67e22;
}

.custom-value {
  color: var(--grafana-warning);
}

.ai-value {
  color: var(--grafana-text-accent);
}

.parameter-control-mode {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.mode-indicator {
  padding: 2px 8px;
  border-radius: var(--border-radius-sm);
  font-size: 0.8em;
  font-weight: 600;
}

.ai-mode {
  background: rgba(50, 208, 255, 0.2);
  color: var(--grafana-text-accent);
}

.custom-mode {
  background: rgba(237, 137, 54, 0.2);
  color: var(--grafana-warning);
}

.parameter-suggestion {
  font-size: 0.8em;
  color: var(--grafana-text-accent);
  text-align: right;
}

.parameter-user {
  font-size: 0.8em;
  color: var(--grafana-warning);
  text-align: right;
  font-weight: 600;
}

.parameter-actions {
  margin-top: var(--space-3);
  text-align: center;
}

.risk-controls {
  display: flex;
  gap: var(--space-2);
}

.risk-btn {
  padding: 4px 8px;
  border: 1px solid var(--grafana-panel-border);
  background: var(--grafana-dark);
  color: var(--grafana-text-secondary);
  border-radius: var(--border-radius-sm);
  font-size: 0.8em;
  cursor: pointer;
  transition: all 0.2s ease;
}

.risk-btn.active {
  background: var(--grafana-text-accent);
  color: var(--grafana-dark);
  border-color: var(--grafana-text-accent);
}

.risk-btn:hover:not(.active) {
  background: rgba(50, 208, 255, 0.1);
  color: var(--grafana-text-accent);
}

.allocation-presets {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.preset-label {
  font-size: 0.9em;
  color: var(--grafana-text-secondary);
}

.preset-btn {
  padding: 2px 6px;
  border: 1px solid var(--grafana-panel-border);
  background: var(--grafana-dark);
  color: var(--grafana-text-secondary);
  border-radius: var(--border-radius-sm);
  font-size: 0.8em;
  cursor: pointer;
  transition: all 0.2s ease;
}

.preset-btn:hover {
  background: rgba(50, 208, 255, 0.1);
  color: var(--grafana-text-accent);
}

.allocation-control {
  display: flex;
  align-items: center;
  gap: var(--space-3);
}

.allocation-input {
  width: 60px;
  padding: 4px 8px;
  border: 1px solid var(--grafana-panel-border);
  background: var(--grafana-dark);
  color: var(--grafana-text-primary);
  border-radius: var(--border-radius-sm);
  text-align: center;
}

.allocation-suggestion {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.suggestion-label {
  font-size: 0.8em;
  color: var(--grafana-text-accent);
}

.user-label {
  font-size: 0.8em;
  color: var(--grafana-warning);
  font-weight: 600;
}

.suggestion-btn {
  background: rgba(50, 208, 255, 0.1);
  color: var(--grafana-text-accent);
  border: 1px solid var(--grafana-text-accent);
  padding: 2px 6px;
  border-radius: var(--border-radius-sm);
  font-size: 0.7em;
  cursor: pointer;
  opacity: 0;
  transition: all 0.2s ease;
}

.suggestion-btn.visible {
  opacity: 1;
}

.suggestion-btn:hover {
  background: var(--grafana-text-accent);
  color: var(--grafana-dark);
}

.allocation-summary {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: var(--space-4);
  padding-top: var(--space-3);
  border-top: 1px solid var(--grafana-panel-border);
}

.total-allocation {
  font-weight: 600;
}

.total-allocation.valid {
  color: var(--grafana-success);
}

.total-allocation.invalid {
  color: var(--grafana-warning);
}

.allocation-actions {
  display: flex;
  gap: var(--space-3);
}

.setting-recommendation {
  font-size: 0.8em;
  color: var(--grafana-text-accent);
  margin-left: var(--space-2);
}

.settings-actions {
  margin-top: var(--space-3);
  text-align: center;
}

.deployment-type {
  text-align: center;
  padding: var(--space-2) var(--space-4);
  border-radius: var(--border-radius);
  margin-bottom: var(--space-4);
  font-weight: 600;
}

.ai-deployment {
  background: rgba(50, 208, 255, 0.1);
  color: var(--grafana-text-accent);
  border: 1px solid var(--grafana-text-accent);
}

.custom-deployment {
  background: rgba(237, 137, 54, 0.1);
  color: var(--grafana-warning);
  border: 1px solid var(--grafana-warning);
}

.banner-details {
  display: flex;
  gap: var(--space-4);
  margin-top: var(--space-2);
}

.detail-item {
  font-size: 0.9em;
  color: var(--grafana-text-secondary);
}

.grafana-btn-text {
  background: transparent;
  border: 1px solid transparent;
  color: var(--grafana-text-secondary);
  padding: var(--space-2) var(--space-4);
  border-radius: var(--border-radius);
  cursor: pointer;
  transition: all 0.2s ease;
}

.grafana-btn-text:hover {
  color: var(--grafana-text-accent);
  background: rgba(50, 208, 255, 0.1);
}
</style>
