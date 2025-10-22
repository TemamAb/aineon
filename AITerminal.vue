<!-- PLATINUM SOURCES: Monaco Editor, CodeMirror -->
<!-- CONTINUAL LEARNING: Query pattern learning, interface optimization -->

<template>
  <div class="ai-terminal">
    <!-- Terminal Header -->
    <div class="terminal-header">
      <div class="terminal-info">
        <h2>AI Trading Intelligence</h2>
        <status-indicator 
          :status="aiStatus"
          :message="statusMessage"
        />
      </div>
      <div class="terminal-controls">
        <model-selector 
          v-model="selectedModel"
          :models="availableModels"
          @model-change="handleModelChange"
        />
        <connection-controls
          :connected="websocketConnected"
          @reconnect="handleReconnect"
        />
      </div>
    </div>

    <!-- Main Terminal Layout -->
    <div class="terminal-layout">
      <!-- AI Response Display -->
      <div class="response-panel">
        <div class="response-header">
          <h3>AI Analysis & Recommendations</h3>
          <div class="response-controls">
            <export-controls 
              :content="currentResponse"
              format="markdown"
            />
            <response-style-selector
              v-model="responseStyle"
              @style-change="handleResponseStyleChange"
            />
          </div>
        </div>
        
        <div class="response-content" :class="`style-${responseStyle}`">
          <ai-response-renderer
            :content="currentResponse"
            :type="responseType"
            :loading="responseLoading"
            @interaction="handleResponseInteraction"
          />
        </div>
      </div>

      <!-- Input and Interaction Panel -->
      <div class="interaction-panel">
        <!-- Query Input -->
        <div class="query-section">
          <code-editor
            v-model="currentQuery"
            :language="queryLanguage"
            :suggestions="querySuggestions"
            @execute="executeQuery"
            @suggestion-select="handleSuggestionSelect"
            class="query-editor"
          />
          
          <div class="query-actions">
            <button 
              class="btn btn-primary"
              :disabled="!canExecuteQuery"
              @click="executeQuery"
            >
              Execute Query
            </button>
            <button 
              class="btn btn-secondary"
              @click="showQueryTemplates"
            >
              Templates
            </button>
            <button 
              class="btn btn-outline"
              @click="clearQuery"
            >
              Clear
            </button>
          </div>
        </div>

        <!-- Quick Actions -->
        <div class="quick-actions">
          <h4>Quick Analysis</h4>
          <div class="action-buttons">
            <quick-action-button
              v-for="action in quickActions"
              :key="action.id"
              :action="action"
              @click="executeQuickAction(action)"
            />
          </div>
        </div>

        <!-- Conversation History -->
        <div class="history-section">
          <conversation-history
            :conversations="conversationHistory"
            :selected-conversation="selectedConversation"
            @conversation-select="selectConversation"
            @conversation-delete="deleteConversation"
          />
        </div>
      </div>
    </div>

    <!-- Advanced Analysis Panel -->
    <div class="advanced-panel" v-if="showAdvancedPanel">
      <advanced-analysis
        :query="currentQuery"
        :response="currentResponse"
        :model="selectedModel"
        @analysis-complete="handleAdvancedAnalysis"
      />
    </div>

    <!-- Query Templates Modal -->
    <modal 
      v-model:visible="showTemplatesModal"
      title="Query Templates"
      width="800"
    >
      <query-templates
        @template-select="applyQueryTemplate"
        @template-create="createQueryTemplate"
      />
    </modal>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, onMounted, onUnmounted } from 'vue';
import { useAITerminal } from '../composables/useAITerminal';
import { useConversationHistory } from '../composables/useConversationHistory';
import { useQueryLearning } from '../composables/useQueryLearning';

// Components
import StatusIndicator from '../components/AI/StatusIndicator.vue';
import ModelSelector from '../components/AI/ModelSelector.vue';
import ConnectionControls from '../components/AI/ConnectionControls.vue';
import CodeEditor from '../components/Editor/CodeEditor.vue';
import AIResponseRenderer from '../components/AI/AIResponseRenderer.vue';
import QuickActionButton from '../components/AI/QuickActionButton.vue';
import ConversationHistory from '../components/AI/ConversationHistory.vue';
import AdvancedAnalysis from '../components/AI/AdvancedAnalysis.vue';
import QueryTemplates from '../components/AI/QueryTemplates.vue';
import ExportControls from '../components/UI/ExportControls.vue';
import ResponseStyleSelector from '../components/AI/ResponseStyleSelector.vue';
import Modal from '../components/UI/Modal.vue';

// Types
interface AIModel {
  id: string;
  name: string;
  provider: string;
  capabilities: string[];
  maxTokens: number;
}

interface QuickAction {
  id: string;
  label: string;
  description: string;
  query: string;
  icon: string;
}

export default defineComponent({
  name: 'AITerminal',
  components: {
    StatusIndicator,
    ModelSelector,
    ConnectionControls,
    CodeEditor,
    AIResponseRenderer,
    QuickActionButton,
    ConversationHistory,
    AdvancedAnalysis,
    QueryTemplates,
    ExportControls,
    ResponseStyleSelector,
    Modal,
  },
  setup() {
    // Composables
    const {
      aiStatus,
      statusMessage,
      websocketConnected,
      currentResponse,
      responseType,
      responseLoading,
      selectedModel,
      availableModels,
      executeAIQuery,
      reconnect,
    } = useAITerminal();

    const {
      conversationHistory,
      selectedConversation,
      selectConversation,
      deleteConversation,
      addToHistory,
    } = useConversationHistory();

    const {
      querySuggestions,
      learnFromQuery,
      getQueryTemplates,
    } = useQueryLearning();

    // Reactive state
    const currentQuery = ref('');
    const responseStyle = ref('standard');
    const showAdvancedPanel = ref(false);
    const showTemplatesModal = ref(false);

    // Computed properties
    const canExecuteQuery = computed(() => 
      currentQuery.value.trim().length > 0 && !responseLoading.value
    );

    const queryLanguage = computed(() => 'plaintext'); // Could be SQL, Python, etc.

    const quickActions = computed<QuickAction[]>(() => [
      {
        id: 'market-analysis',
        label: 'Market Analysis',
        description: 'Get current market conditions and trends',
        query: 'Analyze current market conditions and identify potential trading opportunities.',
        icon: 'í³Š',
      },
      {
        id: 'strategy-review',
        label: 'Strategy Review',
        description: 'Review active trading strategies performance',
        query: 'Review performance of active trading strategies and suggest optimizations.',
        icon: 'í¾¯',
      },
      {
        id: 'risk-assessment',
        label: 'Risk Assessment',
        description: 'Assess current portfolio risk levels',
        query: 'Assess current portfolio risk and provide risk management recommendations.',
        icon: 'í»¡ï¸',
      },
      {
        id: 'arbitrage-opportunities',
        label: 'Arbitrage Scan',
        description: 'Scan for cross-chain arbitrage opportunities',
        query: 'Scan for arbitrage opportunities across different chains and protocols.',
        icon: 'âš¡',
      },
    ]);

    // Methods
    const handleModelChange = (model: AIModel) => {
      selectedModel.value = model;
      // Track model selection for learning
      learnFromQuery('model_selection', { modelId: model.id });
    };

    const executeQuery = async () => {
      if (!canExecuteQuery.value) return;

      const query = currentQuery.value;
      const response = await executeAIQuery(query, selectedModel.value.id);
      
      // Add to conversation history
      addToHistory({
        query,
        response,
        model: selectedModel.value.id,
        timestamp: Date.now(),
      });

      // Learn from query pattern
      learnFromQuery('user_query', {
        query,
        model: selectedModel.value.id,
        responseType: response.type,
      });
    };

    const executeQuickAction = (action: QuickAction) => {
      currentQuery.value = action.query;
      executeQuery();
      
      // Track quick action usage
      learnFromQuery('quick_action', {
        actionId: action.id,
        actionLabel: action.label,
      });
    };

    const handleSuggestionSelect = (suggestion: string) => {
      currentQuery.value = suggestion;
    };

    const clearQuery = () => {
      currentQuery.value = '';
    };

    const handleReconnect = () => {
      reconnect();
    };

    const handleResponseInteraction = (interaction: any) => {
      // Handle interactions with AI response (e.g., clicking on recommendations)
      console.log('Response interaction:', interaction);
      
      // Learn from user interactions
      learnFromQuery('response_interaction', interaction);
    };

    const handleResponseStyleChange = (style: string) => {
      responseStyle.value = style;
    };

    const showQueryTemplates = () => {
      showTemplatesModal.value = true;
    };

    const applyQueryTemplate = (template: any) => {
      currentQuery.value = template.query;
      showTemplatesModal.value = false;
      executeQuery();
    };

    const createQueryTemplate = (template: any) => {
      // Save new template
      learnFromQuery('template_creation', template);
    };

    const handleAdvancedAnalysis = (analysis: any) => {
      // Handle advanced analysis results
      console.log('Advanced analysis:', analysis);
    };

    // Lifecycle
    onMounted(() => {
      // Load initial state
      getQueryTemplates();
    });

    onUnmounted(() => {
      // Cleanup
    });

    return {
      // State
      currentQuery,
      responseStyle,
      showAdvancedPanel,
      showTemplatesModal,
      
      // Computed
      aiStatus,
      statusMessage,
      websocketConnected,
      currentResponse,
      responseType,
      responseLoading,
      selectedModel,
      availableModels,
      canExecuteQuery,
      queryLanguage,
      querySuggestions,
      quickActions,
      conversationHistory,
      selectedConversation,
      
      // Methods
      handleModelChange,
      executeQuery,
      executeQuickAction,
      handleSuggestionSelect,
      clearQuery,
      handleReconnect,
      handleResponseInteraction,
      handleResponseStyleChange,
      showQueryTemplates,
      applyQueryTemplate,
      createQueryTemplate,
      handleAdvancedAnalysis,
      selectConversation,
      deleteConversation,
    };
  },
});
</script>

<style scoped>
.ai-terminal {
  padding: 1rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.terminal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
}

.terminal-info {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.terminal-controls {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.terminal-layout {
  display: grid;
  grid-template-columns: 1fr 400px;
  gap: 1rem;
  flex: 1;
  min-height: 0;
}

.response-panel {
  display: flex;
  flex-direction: column;
  background: var(--background-paper);
  border-radius: 8px;
  overflow: hidden;
}

.response-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-bottom: 1px solid var(--border-color);
}

.response-controls {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.response-content {
  flex: 1;
  padding: 1rem;
  overflow-y: auto;
}

.interaction-panel {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.query-section {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.query-editor {
  height: 150px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
}

.query-actions {
  display: flex;
  gap: 0.5rem;
  justify-content: flex-end;
}

.quick-actions {
  background: var(--background-paper);
  border-radius: 8px;
  padding: 1rem;
}

.action-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.history-section {
  background: var(--background-paper);
  border-radius: 8px;
  padding: 1rem;
  flex: 1;
  overflow-y: auto;
}

.advanced-panel {
  background: var(--background-paper);
  border-radius: 8px;
  padding: 1rem;
  margin-top: 1rem;
}

/* Response style variants */
.style-minimal .response-content {
  padding: 0;
  background: transparent;
}

.style-detailed .response-content {
  background: var(--background-default);
  border: 1px solid var(--border-color);
}

/* Responsive design */
@media (max-width: 1024px) {
  .terminal-layout {
    grid-template-columns: 1fr;
    grid-template-rows: 1fr auto;
  }
  
  .interaction-panel {
    max-height: 400px;
  }
}

@media (max-width: 768px) {
  .terminal-header {
    flex-direction: column;
    align-items: stretch;
  }
  
  .terminal-controls {
    justify-content: space-between;
  }
  
  .action-buttons {
    grid-template-columns: 1fr;
  }
  
  .query-actions {
    flex-direction: column;
  }
}
</style>
