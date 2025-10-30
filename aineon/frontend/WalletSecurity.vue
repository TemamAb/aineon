<!-- PLATINUM SOURCES: Gnosis Safe UI, Argent -->
<!-- CONTINUAL LEARNING: Security behavior learning, access pattern optimization -->

<template>
  <div class="wallet-security">
    <!-- Security Overview Header -->
    <div class="security-header">
      <h2>Wallet Security Dashboard</h2>
      <security-score 
        :score="securityScore"
        :trend="securityTrend"
        :last-updated="lastSecurityUpdate"
      />
    </div>

    <!-- Multi-signature Wallet Management -->
    <div class="multi-sig-section">
      <div class="section-header">
        <h3>Multi-signature Wallets</h3>
        <button 
          class="btn btn-primary"
          @click="showCreateWalletModal = true"
        >
          Create New Wallet
        </button>
      </div>
      
      <div class="wallets-grid">
        <multi-sig-wallet-card
          v-for="wallet in multiSigWallets"
          :key="wallet.address"
          :wallet="wallet"
          @wallet-select="selectWallet"
          @settings-click="showWalletSettings"
        />
      </div>
    </div>

    <!-- Security Monitoring -->
    <div class="security-monitoring">
      <div class="monitoring-grid">
        <!-- Recent Activities -->
        <security-activity-feed
          :activities="recentActivities"
          :loading="activitiesLoading"
          @activity-review="reviewActivity"
          class="activity-feed"
        />
        
        <!-- Access Control -->
        <access-control-panel
          :wallets="multiSigWallets"
          :selected-wallet="selectedWallet"
          :access-rules="accessRules"
          @rule-update="updateAccessRule"
          @emergency-lock="triggerEmergencyLock"
          class="access-panel"
        />
      </div>
    </div>

    <!-- Transaction Queue -->
    <div class="transaction-queue" v-if="pendingTransactions.length > 0">
      <h3>Pending Transactions</h3>
      <transaction-queue
        :transactions="pendingTransactions"
        :selected-wallet="selectedWallet"
        @transaction-approve="approveTransaction"
        @transaction-reject="rejectTransaction"
        @transaction-cancel="cancelTransaction"
      />
    </div>

    <!-- Security Analytics -->
    <div class="security-analytics">
      <analytics-dashboard
        :metrics="securityMetrics"
        :time-range="analyticsTimeRange"
        @time-range-change="updateAnalyticsTimeRange"
      />
    </div>

    <!-- Modals -->
    <create-wallet-modal
      v-model:visible="showCreateWalletModal"
      @wallet-created="handleWalletCreated"
    />
    
    <wallet-settings-modal
      v-model:visible="showWalletSettingsModal"
      :wallet="selectedWallet"
      @settings-updated="handleSettingsUpdated"
    />
    
    <emergency-lock-modal
      v-model:visible="showEmergencyLockModal"
      :wallet="selectedWallet"
      @emergency-locked="handleEmergencyLocked"
    />
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, onMounted } from 'vue';
import { useWalletSecurity } from '../composables/useWalletSecurity';
import { useMultiSig } from '../composables/useMultiSig';
import { useSecurityAnalytics } from '../composables/useSecurityAnalytics';

// Components
import SecurityScore from '../components/Security/SecurityScore.vue';
import MultiSigWalletCard from '../components/Wallet/MultiSigWalletCard.vue';
import SecurityActivityFeed from '../components/Security/SecurityActivityFeed.vue';
import AccessControlPanel from '../components/Security/AccessControlPanel.vue';
import TransactionQueue from '../components/Transactions/TransactionQueue.vue';
import AnalyticsDashboard from '../components/Analytics/AnalyticsDashboard.vue';
import CreateWalletModal from '../components/Wallet/CreateWalletModal.vue';
import WalletSettingsModal from '../components/Wallet/WalletSettingsModal.vue';
import EmergencyLockModal from '../components/Security/EmergencyLockModal.vue';

// Types
interface MultiSigWallet {
  address: string;
  name: string;
  threshold: number;
  owners: string[];
  balance: string;
  pendingTransactions: number;
  securityLevel: 'high' | 'medium' | 'low';
}

interface SecurityActivity {
  id: string;
  type: 'transaction' | 'access' | 'security' | 'approval';
  description: string;
  timestamp: Date;
  severity: 'info' | 'warning' | 'critical';
  walletAddress: string;
}

export default defineComponent({
  name: 'WalletSecurity',
  components: {
    SecurityScore,
    MultiSigWalletCard,
    SecurityActivityFeed,
    AccessControlPanel,
    TransactionQueue,
    AnalyticsDashboard,
    CreateWalletModal,
    WalletSettingsModal,
    EmergencyLockModal,
  },
  setup() {
    // Composables
    const {
      securityScore,
      securityTrend,
      lastSecurityUpdate,
      securityMetrics,
      recentActivities,
      activitiesLoading,
      refreshSecurityData,
    } = useWalletSecurity();

    const {
      multiSigWallets,
      selectedWallet,
      pendingTransactions,
      accessRules,
      createMultiSigWallet,
      approveTransaction,
      rejectTransaction,
      cancelTransaction,
      updateWalletSettings,
      triggerEmergencyLock,
    } = useMultiSig();

    const {
      analyticsTimeRange,
      updateTimeRange,
    } = useSecurityAnalytics();

    // Reactive state
    const showCreateWalletModal = ref(false);
    const showWalletSettingsModal = ref(false);
    const showEmergencyLockModal = ref(false);

    // Computed properties
    const hasCriticalAlerts = computed(() =>
      recentActivities.value.some(activity => activity.severity === 'critical')
    );

    // Methods
    const selectWallet = (wallet: MultiSigWallet) => {
      selectedWallet.value = wallet;
    };

    const showWalletSettings = (wallet: MultiSigWallet) => {
      selectedWallet.value = wallet;
      showWalletSettingsModal.value = true;
    };

    const handleWalletCreated = async (walletConfig: any) => {
      try {
        await createMultiSigWallet(walletConfig);
        showCreateWalletModal.value = false;
        refreshSecurityData();
      } catch (error) {
        console.error('Failed to create wallet:', error);
      }
    };

    const handleSettingsUpdated = async (settings: any) => {
      if (selectedWallet.value) {
        await updateWalletSettings(selectedWallet.value.address, settings);
        showWalletSettingsModal.value = false;
        refreshSecurityData();
      }
    };

    const handleEmergencyLocked = () => {
      showEmergencyLockModal.value = false;
      refreshSecurityData();
    };

    const reviewActivity = (activity: SecurityActivity) => {
      // Handle activity review
      console.log('Review activity:', activity);
    };

    const updateAccessRule = async (rule: any) => {
      await updateWalletSettings(selectedWallet.value!.address, { accessRules: rule });
    };

    const updateAnalyticsTimeRange = (range: string) => {
      updateTimeRange(range);
    };

    // Lifecycle
    onMounted(() => {
      refreshSecurityData();
    });

    return {
      // State
      showCreateWalletModal,
      showWalletSettingsModal,
      showEmergencyLockModal,
      
      // Computed
      securityScore,
      securityTrend,
      lastSecurityUpdate,
      securityMetrics,
      recentActivities,
      activitiesLoading,
      multiSigWallets,
      selectedWallet,
      pendingTransactions,
      accessRules,
      analyticsTimeRange,
      hasCriticalAlerts,
      
      // Methods
      selectWallet,
      showWalletSettings,
      handleWalletCreated,
      handleSettingsUpdated,
      handleEmergencyLocked,
      reviewActivity,
      updateAccessRule,
      approveTransaction,
      rejectTransaction,
      cancelTransaction,
      triggerEmergencyLock,
      updateAnalyticsTimeRange,
    };
  },
});
</script>

<style scoped>
.wallet-security {
  padding: 1rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.security-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.wallets-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1rem;
}

.security-monitoring {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.monitoring-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 1rem;
}

.activity-feed,
.access-panel {
  background: var(--background-paper);
  border-radius: 8px;
  padding: 1rem;
}

.transaction-queue {
  background: var(--background-paper);
  border-radius: 8px;
  padding: 1rem;
}

.security-analytics {
  background: var(--background-paper);
  border-radius: 8px;
  padding: 1rem;
}

/* Responsive design */
@media (max-width: 1024px) {
  .monitoring-grid {
    grid-template-columns: 1fr;
  }
  
  .wallets-grid {
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  }
}

@media (max-width: 768px) {
  .security-header {
    flex-direction: column;
    align-items: stretch;
  }
  
  .section-header {
    flex-direction: column;
    align-items: stretch;
    gap: 1rem;
  }
  
  .wallets-grid {
    grid-template-columns: 1fr;
  }
}

/* Critical alert styling */
.critical-alert {
  border-left: 4px solid var(--error-color);
  background: color-mix(in srgb, var(--error-color) 10%, transparent);
}
</style>
