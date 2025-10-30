<!-- PLATINUM SOURCES: Ant Design, Element UI -->
<!-- CONTINUAL LEARNING: Parameter tuning patterns, user preference learning -->

<template>
  <div class="trading-parameters">
    <!-- Strategy Selection and Management -->
    <div class="strategy-header">
      <h2>Strategy Configuration</h2>
      <div class="strategy-controls">
        <strategy-selector 
          v-model="selectedStrategy"
          :strategies="availableStrategies"
          @strategy-change="handleStrategyChange"
        />
        <button 
          class="btn btn-primary"
          @click="showSaveTemplateModal = true"
        >
          Save as Template
        </button>
        <button 
          class="btn btn-secondary"
          @click="loadRecommendedParameters"
        >
          AI Recommendations
        </button>
      </div>
    </div>

    <!-- Parameter Configuration Grid -->
    <div class="parameters-grid">
      <!-- Basic Parameters -->
      <parameter-section title="Basic Configuration" :default-open="true">
        <parameter-input
          v-for="param in basicParameters"
          :key="param.name"
          v-model="parameters[param.name]"
          :definition="param"
          @change="handleParameterChange(param.name, $event)"
        />
      </parameter-section>

      <!-- Risk Management -->
      <parameter-section title="Risk Management">
        <risk-parameters
          :parameters="riskParameters"
          :current-values="parameters"
          @parameter-change="handleRiskParameterChange"
        />
        <risk-visualization
          :parameters="parameters"
          :risk-metrics="calculatedRiskMetrics"
          class="risk-viz"
        />
      </parameter-section>

      <!-- Advanced Strategy Parameters -->
      <parameter-section title="Advanced Parameters">
        <advanced-parameters
          :strategy-type="selectedStrategy.type"
          :parameters="advancedParameters"
          :current-values="parameters"
          @parameter-change="handleAdvancedParameterChange"
        />
      </parameter-section>

      <!-- Backtesting Results -->
      <parameter-section title="Performance Analysis">
        <backtest-results
          :strategy="selectedStrategy"
          :parameters="parameters"
          :results="backtestResults"
          :loading="backtestLoading"
          @run-backtest="runBacktest"
        />
        <parameter-optimization
          :strategy="selectedStrategy"
          :current-parameters="parameters"
          :optimization-results="optimizationResults"
          @run-optimization="runParameterOptimization"
        />
      </parameter-section>
    </div>

    <!-- Action Buttons -->
    <div class="action-buttons">
      <button 
        class="btn btn-success"
        :disabled="!parametersValid"
        @click="deployStrategy"
      >
        Deploy Strategy
      </button>
      <button 
        class="btn btn-warning"
        @click="resetToDefaults"
      >
        Reset to Defaults
      </button>
      <button 
        class="btn btn-info"
        @click="showParameterHistory"
      >
        Parameter History
      </button>
    </div>

    <!-- Save Template Modal -->
    <modal 
      v-model:visible="showSaveTemplateModal"
      title="Save Parameter Template"
      @ok="saveParameterTemplate"
      @cancel="showSaveTemplateModal = false"
    >
      <template-form
        :parameters="parameters"
        :strategy="selectedStrategy"
        @template-saved="handleTemplateSaved"
      />
    </modal>

    <!-- Parameter History Drawer -->
    <drawer
      v-model:visible="showHistoryDrawer"
      title="Parameter Change History"
      placement="right"
      width="400"
    >
      <parameter-history
        :strategy-id="selectedStrategy.id"
        @parameter-restore="handleParameterRestore"
      />
    </drawer>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, watch } from 'vue';
import { useStrategyStore } from '../stores/strategyStore';
import { useParameterLearning } from '../composables/useParameterLearning';
import { useBacktesting } from '../composables/useBacktesting';

// Components
import StrategySelector from '../components/Strategy/StrategySelector.vue';
import ParameterSection from '../components/Parameters/ParameterSection.vue';
import ParameterInput from '../components/Parameters/ParameterInput.vue';
import RiskParameters from '../components/Parameters/RiskParameters.vue';
import RiskVisualization from '../components/Visualization/RiskVisualization.vue';
import AdvancedParameters from '../components/Parameters/AdvancedParameters.vue';
import BacktestResults from '../components/Backtesting/BacktestResults.vue';
import ParameterOptimization from '../components/Optimization/ParameterOptimization.vue';
import TemplateForm from '../components/Templates/TemplateForm.vue';
import ParameterHistory from '../components/History/ParameterHistory.vue';
import Modal from '../components/UI/Modal.vue';
import Drawer from '../components/UI/Drawer.vue';

// Types
interface Strategy {
  id: string;
  name: string;
  type: string;
  category: string;
  defaultParameters: Record<string, any>;
}

interface ParameterDefinition {
  name: string;
  type: 'number' | 'string' | 'boolean' | 'select';
  label: string;
  description: string;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ label: string; value: any }>;
  required: boolean;
}

export default defineComponent({
  name: 'TradingParameters',
  components: {
    StrategySelector,
    ParameterSection,
    ParameterInput,
    RiskParameters,
    RiskVisualization,
    AdvancedParameters,
    BacktestResults,
    ParameterOptimization,
    TemplateForm,
    ParameterHistory,
    Modal,
    Drawer,
  },
  setup() {
    // Stores and composables
    const strategyStore = useStrategyStore();
    const { learnFromParameterChanges, getRecommendedParameters } = useParameterLearning();
    const { runBacktest, backtestResults, backtestLoading } = useBacktesting();

    // Reactive state
    const selectedStrategy = ref<Strategy>(strategyStore.availableStrategies[0]);
    const parameters = ref<Record<string, any>>({});
    const optimizationResults = ref<any[]>([]);
    const showSaveTemplateModal = ref(false);
    const showHistoryDrawer = ref(false);

    // Computed properties
    const availableStrategies = computed(() => strategyStore.availableStrategies);
    
    const basicParameters = computed<ParameterDefinition[]>(() => 
      strategyStore.getParameterDefinitions(selectedStrategy.value.id, 'basic')
    );

    const riskParameters = computed<ParameterDefinition[]>(() =>
      strategyStore.getParameterDefinitions(selectedStrategy.value.id, 'risk')
    );

    const advancedParameters = computed<ParameterDefinition[]>(() =>
      strategyStore.getParameterDefinitions(selectedStrategy.value.id, 'advanced')
    );

    const calculatedRiskMetrics = computed(() => 
      strategyStore.calculateRiskMetrics(parameters.value)
    );

    const parametersValid = computed(() =>
      strategyStore.validateParameters(parameters.value)
    );

    // Methods
    const handleStrategyChange = (strategy: Strategy) => {
      selectedStrategy.value = strategy;
      // Load strategy-specific parameters
      parameters.value = { ...strategy.defaultParameters };
      
      // Load learned parameter preferences
      const recommended = getRecommendedParameters(strategy.id);
      if (recommended) {
        parameters.value = { ...parameters.value, ...recommended };
      }

      // Track strategy selection for learning
      learnFromParameterChanges('strategy_selection', { strategyId: strategy.id });
    };

    const handleParameterChange = (paramName: string, value: any) => {
      parameters.value[paramName] = value;
      
      // Learn from parameter changes
      learnFromParameterChanges('parameter_adjustment', {
        strategyId: selectedStrategy.value.id,
        parameter: paramName,
        value: value,
        timestamp: Date.now(),
      });
    };

    const handleRiskParameterChange = (updates: Record<string, any>) => {
      Object.assign(parameters.value, updates);
    };

    const handleAdvancedParameterChange = (updates: Record<string, any>) => {
      Object.assign(parameters.value, updates);
    };

    const deployStrategy = async () => {
      try {
        await strategyStore.deployStrategy(
          selectedStrategy.value.id,
          parameters.value
        );
        // Track successful deployment
        learnFromParameterChanges('strategy_deployment', {
          strategyId: selectedStrategy.value.id,
          parameters: parameters.value,
          success: true,
        });
      } catch (error) {
        learnFromParameterChanges('strategy_deployment', {
          strategyId: selectedStrategy.value.id,
          parameters: parameters.value,
          success: false,
          error: error.message,
        });
        throw error;
      }
    };

    const loadRecommendedParameters = async () => {
      const recommendations = await strategyStore.getAIRecommendations(
        selectedStrategy.value.id
      );
      if (recommendations) {
        parameters.value = { ...parameters.value, ...recommendations };
      }
    };

    const resetToDefaults = () => {
      parameters.value = { ...selectedStrategy.value.defaultParameters };
    };

    const saveParameterTemplate = async (templateData: any) => {
      await strategyStore.saveParameterTemplate(templateData);
      showSaveTemplateModal.value = false;
    };

    const handleTemplateSaved = () => {
      showSaveTemplateModal.value = false;
    };

    const showParameterHistory = () => {
      showHistoryDrawer.value = true;
    };

    const handleParameterRestore = (historicalParams: Record<string, any>) => {
      parameters.value = { ...parameters.value, ...historicalParams };
      showHistoryDrawer.value = false;
    };

    const runParameterOptimization = async () => {
      optimizationResults.value = await strategyStore.optimizeParameters(
        selectedStrategy.value.id,
        parameters.value
      );
    };

    // Initialize
    handleStrategyChange(selectedStrategy.value);

    return {
      // State
      selectedStrategy,
      parameters,
      optimizationResults,
      showSaveTemplateModal,
      showHistoryDrawer,
      backtestResults,
      backtestLoading,
      
      // Computed
      availableStrategies,
      basicParameters,
      riskParameters,
      advancedParameters,
      calculatedRiskMetrics,
      parametersValid,
      
      // Methods
      handleStrategyChange,
      handleParameterChange,
      handleRiskParameterChange,
      handleAdvancedParameterChange,
      deployStrategy,
      loadRecommendedParameters,
      resetToDefaults,
      saveParameterTemplate,
      handleTemplateSaved,
      showParameterHistory,
      handleParameterRestore,
      runBacktest,
      runParameterOptimization,
    };
  },
});
</script>

<style scoped>
.trading-parameters {
  padding: 1rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.strategy-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
}

.strategy-controls {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
}

.parameters-grid {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  flex: 1;
  overflow-y: auto;
}

.action-buttons {
  display: flex;
  gap: 0.5rem;
  justify-content: flex-end;
  padding-top: 1rem;
  border-top: 1px solid var(--border-color);
}

.risk-viz {
  margin-top: 1rem;
  height: 200px;
}

/* Responsive design */
@media (max-width: 768px) {
  .strategy-header {
    flex-direction: column;
    align-items: stretch;
  }
  
  .strategy-controls {
    justify-content: space-between;
  }
  
  .action-buttons {
    flex-direction: column;
  }
}

@media (max-width: 480px) {
  .strategy-controls {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
