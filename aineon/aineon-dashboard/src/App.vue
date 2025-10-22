<template>
  <div id="app">
    <!-- Grafana Navigation -->
    <nav class="grafana-nav">
      <div class="nav-header">
        <h2>íº€ Aineon</h2>
      </div>
      <div class="nav-menu">
        <div 
          v-for="module in modules" 
          :key="module.id"
          :class="['nav-item', { active: currentModule === module.id }]"
          @click="switchModule(module.id)"
        >
          <span class="nav-icon">{{ module.icon }}</span>
          <span class="nav-text">{{ module.name }}</span>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="main-content">
      <!-- Grafana Header -->
      <header class="grafana-header">
        <div class="header-left">
          <h1>{{ currentModuleName }}</h1>
        </div>
        <div class="header-right">
          <select v-model="refreshRate" class="grafana-select">
            <option value="1000">1s Refresh</option>
            <option value="3000">3s Refresh</option>
            <option value="5000">5s Refresh</option>
            <option value="10000">10s Refresh</option>
          </select>
          <span class="total-profit">
            í²° ${{ totalProfit.toLocaleString() }}/{{ daysActive }} days
          </span>
          <select v-model="currency" class="grafana-select">
            <option value="USD">USD</option>
            <option value="ETH">ETH</option>
          </select>
          <button v-if="!connected" @click="connectWallet" class="grafana-btn">
            Connect Wallet
          </button>
          <div v-else class="wallet-info">
            <span class="status-indicator status-online"></span>
            {{ shortenedAddress }}
          </div>
        </div>
      </header>

      <!-- Dashboard Grid -->
      <div class="dashboard-grid">
        <component :is="currentModuleComponent" />
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import LiveMonitoring from './components/LiveMonitoring.vue'
import StrategyParameters from './components/StrategyParameters.vue'
import OptimizeSimulate from './components/OptimizeSimulate.vue'
import DeployLive from './components/DeployLive.vue'
import ContinuousOptimization from './components/ContinuousOptimization.vue'
import WithdrawProfit from './components/WithdrawProfit.vue'
import AITerminal from './components/AITerminal.vue'

export default {
  name: 'App',
  components: {
    LiveMonitoring,
    StrategyParameters,
    OptimizeSimulate,
    DeployLive,
    ContinuousOptimization,
    WithdrawProfit,
    AITerminal
  },
  setup() {
    const currentModule = ref('live-monitoring')
    const refreshRate = ref(1000)
    const currency = ref('USD')
    const connected = ref(false)
    const userAddress = ref('')
    const totalProfit = ref(284750)
    const daysActive = ref(47)

    const modules = [
      { id: 'live-monitoring', name: 'Live Monitoring', icon: 'í¿ ' },
      { id: 'strategy-parameters', name: 'Strategy Parameters', icon: 'âš™ï¸' },
      { id: 'optimize-simulate', name: 'Optimize & Simulate', icon: 'í´–' },
      { id: 'deploy-live', name: 'Deploy to Live', icon: 'íº€' },
      { id: 'continuous-optimization', name: 'Continuous Optimization', icon: 'í³ˆ' },
      { id: 'withdraw-profit', name: 'Withdraw Profit', icon: 'í²°' },
      { id: 'ai-terminal', name: 'AI Terminal', icon: 'í· ' }
    ]

    const currentModuleName = computed(() => {
      const module = modules.find(m => m.id === currentModule.value)
      return module ? module.name : 'Dashboard'
    })

    const currentModuleComponent = computed(() => {
      const components = {
        'live-monitoring': LiveMonitoring,
        'strategy-parameters': StrategyParameters,
        'optimize-simulate': OptimizeSimulate,
        'deploy-live': DeployLive,
        'continuous-optimization': ContinuousOptimization,
        'withdraw-profit': WithdrawProfit,
        'ai-terminal': AITerminal
      }
      return components[currentModule.value]
    })

    const shortenedAddress = computed(() => {
      if (!userAddress.value) return ''
      return `${userAddress.value.slice(0, 6)}...${userAddress.value.slice(-4)}`
    })

    const switchModule = (moduleId) => {
      currentModule.value = moduleId
    }

    const connectWallet = async () => {
      if (typeof window.ethereum !== 'undefined') {
        try {
          const accounts = await window.ethereum.request({
            method: 'eth_requestAccounts'
          })
          userAddress.value = accounts[0]
          connected.value = true
          console.log('Wallet connected:', accounts[0])
        } catch (error) {
          console.error('Wallet connection failed:', error)
        }
      } else {
        alert('MetaMask not detected. Please install MetaMask.')
      }
    }

    onMounted(() => {
      // Start live data updates
      setInterval(() => {
        // Simulate live profit updates
        totalProfit.value += Math.random() * 100
      }, refreshRate.value)
    })

    return {
      currentModule,
      refreshRate,
      currency,
      connected,
      userAddress,
      totalProfit,
      daysActive,
      modules,
      currentModuleName,
      currentModuleComponent,
      shortenedAddress,
      switchModule,
      connectWallet
    }
  }
}
</script>

<style>
@import './assets/grafana-theme.css';

#app {
  display: grid;
  grid-template-columns: 250px 1fr;
  min-height: 100vh;
}

.main-content {
  margin-left: 250px;
  min-height: 100vh;
}

.header-right {
  display: flex;
  align-items: center;
  gap: var(--space-4);
}

.total-profit {
  background: rgba(115, 191, 105, 0.2);
  color: var(--grafana-success);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--border-radius);
  font-size: 0.9em;
}

.wallet-info {
  display: flex;
  align-items: center;
  background: rgba(50, 208, 255, 0.1);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--border-radius);
  font-size: 0.9em;
}

@media (max-width: 768px) {
  #app {
    grid-template-columns: 1fr;
  }
  
  .main-content {
    margin-left: 0;
  }
  
  .grafana-nav {
    display: none;
  }
}
</style>
