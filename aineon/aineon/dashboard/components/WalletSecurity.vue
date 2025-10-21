<template>
  <div class="wallet-security-module">
    <div class="module-header">
      <h3>í´’ Wallet & Security</h3>
      <div class="view-toggle" @click="toggleView">
        {{ isAdvancedView ? 'Default' : 'Advanced' }}
      </div>
    </div>
    
    <div v-if="!isAdvancedView" class="default-view">
      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-value">${{ totalBalance }}M</div>
          <div class="metric-label">Total Balance</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{{ activeThreats }}</div>
          <div class="metric-label">Active Threats</div>
        </div>
        <div class="metric-card">
          <div class="metric-value" :class="multiSigStatus">{{ multiSigStatus }}</div>
          <div class="metric-label">Multi-Sig Status</div>
        </div>
      </div>
    </div>

    <div v-else class="advanced-view">
      <div class="advanced-metrics">
        <div class="metric-section">
          <h4>Multi-Chain Balances</h4>
          <div class="chain-balances">
            <div v-for="chain in chainBalances" :key="chain.name" class="chain-item">
              <span class="chain-name">{{ chain.name }}</span>
              <span class="chain-balance">${{ chain.balance }}M</span>
            </div>
          </div>
        </div>
        
        <div class="metric-section">
          <h4>Threat Monitoring</h4>
          <div class="threat-grid">
            <div class="threat-item">
              <span>API Key Rotation</span>
              <span :class="apiKeyStatus">{{ apiKeyStatus }}</span>
            </div>
            <div class="threat-item">
              <span>Connection Health</span>
              <span>{{ connectionHealth }}%</span>
            </div>
            <div class="threat-item">
              <span>Security Scans</span>
              <span>{{ securityScans }}/day</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'WalletSecurity',
  data() {
    return {
      isAdvancedView: false,
      totalBalance: 2.14,
      activeThreats: 0,
      multiSigStatus: 'Active',
      chainBalances: [
        { name: 'Ethereum', balance: 1.2 },
        { name: 'Polygon', balance: 0.4 },
        { name: 'Arbitrum', balance: 0.3 },
        { name: 'BSC', balance: 0.24 }
      ],
      apiKeyStatus: 'Secure',
      connectionHealth: 99.8,
      securityScans: 144
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
.wallet-security-module {
  border: 1px solid #e1e8ed;
  border-radius: 8px;
  padding: 16px;
  background: white;
}

.multiSigStatus.Active { color: #10b981; }
.multiSigStatus.Warning { color: #f59e0b; }
.multiSigStatus.Critical { color: #ef4444; }

.apiKeyStatus.Secure { color: #10b981; }
.apiKeyStatus.Warning { color: #f59e0b; }

.chain-balances {
  display: grid;
  gap: 8px;
}

.chain-item {
  display: flex;
  justify-content: space-between;
  padding: 8px;
  background: #f8fafc;
  border-radius: 4px;
}

.threat-grid {
  display: grid;
  gap: 8px;
}

.threat-item {
  display: flex;
  justify-content: space-between;
  padding: 8px;
  background: #f8fafc;
  border-radius: 4px;
}
</style>
