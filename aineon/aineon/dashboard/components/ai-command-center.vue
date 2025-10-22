<template>
  <div class="ai-command-center">
    <!-- Intelligence Overview -->
    <div class="intelligence-overview">
      <div class="overview-card" v-for="card in overviewCards" :key="card.id" :class="card.status">
        <div class="card-icon">{{ card.icon }}</div>
        <div class="card-content">
          <h3>{{ card.title }}</h3>
          <div class="card-value">{{ card.value }}</div>
          <div class="card-trend" :class="card.trend">{{ card.trendValue }}</div>
        </div>
      </div>
    </div>

    <!-- AI Agent Swarm -->
    <div class="ai-swarm-dashboard">
      <h2>Ì¥ñ AI Agent Swarm Coordination</h2>
      <div class="swarm-grid">
        <div class="agent-card" v-for="agent in aiAgents" :key="agent.id" :class="agent.status">
          <div class="agent-header">
            <span class="agent-icon">{{ agent.icon }}</span>
            <span class="agent-name">{{ agent.name }}</span>
            <span class="agent-status">{{ agent.status }}</span>
          </div>
          <div class="agent-metrics">
            <div class="metric" v-for="metric in agent.metrics" :key="metric.name">
              <span class="metric-name">{{ metric.name }}:</span>
              <span class="metric-value">{{ metric.value }}</span>
            </div>
          </div>
          <div class="agent-decision">
            <strong>Decision:</strong> {{ agent.decision }}
          </div>
        </div>
      </div>
    </div>

    <!-- Real-time Intelligence Fusion -->
    <div class="intelligence-fusion">
      <h2>ÌæØ Cross-Modal Intelligence Fusion</h2>
      <div class="fusion-matrix">
        <div class="modality" v-for="modality in dataModalities" :key="modality.id">
          <div class="modality-header">
            <span class="modality-name">{{ modality.name }}</span>
            <span class="modality-weight">{{ modality.weight }}</span>
          </div>
          <div class="modality-signal" :style="{ width: modality.confidence * 100 + '%' }">
            {{ (modality.confidence * 100).toFixed(1) }}%
          </div>
          <div class="modality-contribution">
            Contribution: {{ (modality.contribution * 100).toFixed(1) }}%
          </div>
        </div>
      </div>
    </div>

    <!-- Metalearning Strategies -->
    <div class="metalearning-strategies">
      <h2>Ì¥ß Metalearning Strategy Generation</h2>
      <div class="strategies-grid">
        <div class="strategy-card" v-for="strategy in generatedStrategies" :key="strategy.id">
          <div class="strategy-header">
            <span class="strategy-name">{{ strategy.name }}</span>
            <span class="strategy-type">{{ strategy.type }}</span>
          </div>
          <div class="strategy-performance">
            <div class="performance-metric">
              <span>Expected Profit:</span>
              <span :class="strategy.profit > 0 ? 'positive' : 'negative'">
                {{ (strategy.profit * 100).toFixed(2) }}%
              </span>
            </div>
            <div class="performance-metric">
              <span>Confidence:</span>
              <span>{{ (strategy.confidence * 100).toFixed(1) }}%</span>
            </div>
          </div>
          <div class="strategy-actions">
            <button @click="deployStrategy(strategy)" class="btn-deploy">Deploy</button>
            <button @click="backtestStrategy(strategy)" class="btn-backtest">Backtest</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Predictive Regime Detection -->
    <div class="regime-prediction">
      <h2>Ìºä Predictive Regime Detection</h2>
      <div class="regime-display">
        <div class="current-regime">
          <strong>Current:</strong> {{ currentRegime.name }}
          <span class="regime-confidence">{{ (currentRegime.confidence * 100).toFixed(1) }}%</span>
        </div>
        <div class="predicted-regime">
          <strong>Predicted (8h):</strong> {{ predictedRegime.name }}
          <span class="prediction-confidence">{{ (predictedRegime.confidence * 100).toFixed(1) }}%</span>
        </div>
        <div class="regime-timeline">
          <div class="timeline-item" v-for="event in regimeTimeline" :key="event.time">
            <span class="time">{{ event.time }}</span>
            <span class="regime">{{ event.regime }}</span>
            <span class="confidence">{{ (event.confidence * 100).toFixed(0) }}%</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'AICommandCenter',
  props: {
    aiContext: Object,
    optimizationStats: Object
  },
  data() {
    return {
      overviewCards: [
        {
          id: 'accuracy',
          title: 'AI Accuracy',
          value: '97.8%',
          trend: 'up',
          trendValue: '+2.7%',
          status: 'optimal',
          icon: 'ÌæØ'
        },
        {
          id: 'regime',
          title: 'Market Regime',
          value: 'High Volatility',
          trend: 'stable',
          trendValue: 'Bullish',
          status: 'active',
          icon: 'Ìºä'
        },
        {
          id: 'optimization',
          title: 'Continuous Opt',
          value: '3min cycles',
          trend: 'up',
          trendValue: '480/day',
          status: 'active',
          icon: '‚ö°'
        },
        {
          id: 'strategies',
          title: 'Active Strategies',
          value: '15+',
          trend: 'up',
          trendValue: 'Auto-gen',
          status: 'optimal',
          icon: 'Ì¥ß'
        }
      ],
      aiAgents: [
        {
          id: 'strategic',
          name: 'Strategic AI',
          icon: 'Ì±ë',
          status: 'active',
          metrics: [
            { name: 'Accuracy', value: '96.2%' },
            { name: 'Confidence', value: '94.8%' },
            { name: 'Regime Score', value: '0.89' }
          ],
          decision: 'Aggressive Long'
        },
        {
          id: 'tactical',
          name: 'Tactical AI',
          icon: 'ÌæØ',
          status: 'active',
          metrics: [
            { name: 'Accuracy', value: '95.7%' },
            { name: 'Signal Strength', value: '0.92' },
            { name: 'Opportunities', value: '24' }
          ],
          decision: 'Momentum Breakout'
        },
        {
          id: 'risk',
          name: 'Risk AI',
          icon: 'Ìª°Ô∏è',
          status: 'active',
          metrics: [
            { name: 'VaR', value: '1.8%' },
            { name: 'Drawdown', value: '6.2%' },
            { name: 'Correlation', value: '0.34' }
          ],
          decision: 'Moderate Exposure'
        },
        {
          id: 'execution',
          name: 'Execution AI',
          icon: '‚ö°',
          status: 'active',
          metrics: [
            { name: 'Speed', value: '48ms' },
            { name: 'Slippage', value: '0.07%' },
            { name: 'Success Rate', value: '99.3%' }
          ],
          decision: 'Stealth Execution'
        }
      ],
      dataModalities: [
        { id: 'price', name: 'Price Data', weight: '0.25', confidence: 0.96, contribution: 0.28 },
        { id: 'volume', name: 'Volume', weight: '0.20', confidence: 0.92, contribution: 0.22 },
        { id: 'sentiment', name: 'Market Sentiment', weight: '0.15', confidence: 0.88, contribution: 0.16 },
        { id: 'onchain', name: 'On-Chain', weight: '0.18', confidence: 0.91, contribution: 0.20 },
        { id: 'options', name: 'Options Flow', weight: '0.12', confidence: 0.85, contribution: 0.12 },
        { id: 'macro', name: 'Macro Data', weight: '0.10', confidence: 0.82, contribution: 0.10 }
      ],
      generatedStrategies: [
        {
          id: 'strat-1',
          name: 'Volatility Surface Arb',
          type: 'Composite',
          profit: 0.0123,
          confidence: 0.89,
          components: ['volatility', 'correlation', 'timing']
        },
        {
          id: 'strat-2',
          name: 'Cross-Chain Triangular',
          type: 'Arbitrage',
          profit: 0.0098,
          confidence: 0.92,
          components: ['multi-chain', 'triangular', 'liquidity']
        },
        {
          id: 'strat-3',
          name: 'Yield + Arbitrage',
          type: 'Composite',
          profit: 0.0115,
          confidence: 0.87,
          components: ['yield-farming', 'arbitrage', 'risk-hedged']
        }
      ],
      currentRegime: {
        name: 'High Volatility',
        confidence: 0.94
      },
      predictedRegime: {
        name: 'Trending Bull',
        confidence: 0.82
      },
      regimeTimeline: [
        { time: 'Now', regime: 'High Volatility', confidence: 0.94 },
        { time: '+2h', regime: 'Volatility Compression', confidence: 0.78 },
        { time: '+4h', regime: 'Breakout Imminent', confidence: 0.85 },
        { time: '+6h', regime: 'Trending Bull', confidence: 0.82 },
        { time: '+8h', regime: 'Momentum Phase', confidence: 0.79 }
      ]
    }
  },
  methods: {
    deployStrategy(strategy) {
      this.$emit('moduleEvent', {
        type: 'strategyGenerated',
        data: strategy
      })
      
      this.$notify({
        title: 'Strategy Deployment',
        message: `Deploying ${strategy.name}`,
        type: 'success'
      })
    },
    
    backtestStrategy(strategy) {
      this.$notify({
        title: 'Backtest Initiated',
        message: `Testing ${strategy.name}`,
        type: 'info'
      })
    }
  }
}
</script>

<style scoped>
.ai-command-center {
  padding: 20px;
  color: white;
}

.intelligence-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.overview-card {
  background: #2d3748;
  padding: 20px;
  border-radius: 10px;
  border-left: 4px solid #48bb78;
}

.overview-card.optimal {
  border-left-color: #48bb78;
}

.overview-card.active {
  border-left-color: #4299e1;
}

.swarm-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.agent-card {
  background: #2d3748;
  padding: 15px;
  border-radius: 8px;
  border: 1px solid #4a5568;
}

.fusion-matrix {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 30px;
}

.modality {
  background: #2d3748;
  padding: 15px;
  border-radius: 8px;
}

.strategies-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 15px;
}

.strategy-card {
  background: #2d3748;
  padding: 15px;
  border-radius: 8px;
  border: 1px solid #4a5568;
}
</style>
