<!-- PLATINUM SOURCES: Yearn Finance UI, Balancer -->
<!-- CONTINUAL LEARNING: Withdrawal pattern learning, fee optimization -->

<template>
  <div class="profit-withdrawal">
    <!-- Profit Overview Header -->
    <div class="profit-header">
      <div class="profit-summary">
        <h2>Profit Management</h2>
        <div class="profit-metrics">
          <metric-card
            title="Total Profits"
            :value="totalProfits"
            value-prefix="$"
            :trend="profitTrend"
            class="total-profit"
          />
          <metric-card
            title="Available for Withdrawal"
            :value="availableForWithdrawal"
            value-prefix="$"
            :trend="withdrawalTrend"
            class="available-profit"
          />
          <metric-card
            title="This Month"
            :value="monthlyProfits"
            value-prefix="$"
            :trend="monthlyTrend"
            class="monthly-profit"
          />
        </div>
      </div>
      <div class="profit-actions">
        <withdrawal-wizard
          :available-amount="availableForWithdrawal"
          @withdrawal-request="initiateWithdrawal"
        />
        <reinvestment-planner
          :available-amount="availableForWithdrawal"
          @reinvestment-plan="showReinvestmentPlan"
        />
      </div>
    </div>

    <!-- Profit Distribution -->
    <div class="distribution-section">
      <h3>Profit Distribution</h3>
      <distribution-chart
        :distribution="profitDistribution"
        :time-range="distributionTimeRange"
        @slice-click="handleDistributionSliceClick"
        class="distribution-chart"
      />
      
      <distribution-settings
        :settings="distributionSettings"
        @settings-update="updateDistributionSettings"
        class="distribution-settings"
      />
    </div>

    <!-- Withdrawal History -->
    <div class="history-section">
      <div class="section-header">
        <h3>Withdrawal History</h3>
        <div class="history-controls">
          <time-range-selector
            v-model="historyTimeRange"
            :ranges="historyRanges"
          />
          <export-button
            :data="withdrawalHistory"
            filename="withdrawal-history"
            format="csv"
          />
        </div>
      </div>
      
      <withdrawal-history-table
        :withdrawals="filteredWithdrawals"
        :loading="historyLoading"
        @withdrawal-detail="showWithdrawalDetail"
        class="history-table"
      />
    </div>

    <!-- Tax and Reporting -->
    <div class="tax-section">
      <h3>Tax Reporting</h3>
      <tax-reporting
        :transactions="taxableTransactions"
        :fiscal-year="currentFiscalYear"
        @report-generate="generateTaxReport"
        @report-export="exportTaxReport"
      />
    </div>

    <!-- Reinvestment Opportunities -->
    <div class="reinvestment-section" v-if="reinvestmentOpportunities.length > 0">
      <h3>Reinvestment Opportunities</h3>
      <opportunity-grid
        :opportunities="reinvestmentOpportunities"
        :available-funds="availableForWithdrawal"
        @opportunity-select="selectReinvestmentOpportunity"
        class="opportunity-grid"
      />
    </div>

    <!-- Modals -->
    <withdrawal-detail-modal
      v-model:visible="showWithdrawalDetailModal"
      :withdrawal="selectedWithdrawal"
    />
    
    <reinvestment-modal
      v-model:visible="showReinvestmentModal"
      :opportunity="selectedOpportunity"
      :amount="reinvestmentAmount"
      @reinvestment-confirm="confirmReinvestment"
    />
    
    <tax-report-modal
      v-model:visible="showTaxReportModal"
      :report="taxReport"
      @report-download="downloadTaxReport"
    />
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, onMounted } from 'vue';
import { useProfitManagement } from '../composables/useProfitManagement';
import { useWithdrawalHistory } from '../composables/useWithdrawalHistory';
import { useTaxReporting } from '../composables/useTaxReporting';
import { useReinvestment } from '../composables/useReinvestment';

// Components
import MetricCard from '../components/Metrics/MetricCard.vue';
import WithdrawalWizard from '../components/Withdrawal/WithdrawalWizard.vue';
import ReinvestmentPlanner from '../components/Reinvestment/ReinvestmentPlanner.vue';
import DistributionChart from '../components/Distribution/DistributionChart.vue';
import DistributionSettings from '../components/Distribution/DistributionSettings.vue';
import WithdrawalHistoryTable from '../components/History/WithdrawalHistoryTable.vue';
import TaxReporting from '../components/Tax/TaxReporting.vue';
import OpportunityGrid from '../components/Reinvestment/OpportunityGrid.vue';
import WithdrawalDetailModal from '../components/Withdrawal/WithdrawalDetailModal.vue';
import ReinvestmentModal from '../components/Reinvestment/ReinvestmentModal.vue';
import TaxReportModal from '../components/Tax/TaxReportModal.vue';
import TimeRangeSelector from '../components/Controls/TimeRangeSelector.vue';
import ExportButton from '../components/UI/ExportButton.vue';

// Types
interface ProfitDistribution {
  category: string;
  amount: number;
  percentage: number;
  color: string;
}

interface Withdrawal {
  id: string;
  timestamp: Date;
  amount: number;
  currency: string;
  status: 'pending' | 'completed' | 'failed';
  transactionHash?: string;
  fee: number;
  destination: string;
}

interface ReinvestmentOpportunity {
  id: string;
  name: string;
  apy: number;
  risk: 'low' | 'medium' | 'high';
  minAmount: number;
  maxAmount: number;
  description: string;
  protocol: string;
}

export default defineComponent({
  name: 'ProfitWithdrawal',
  components: {
    MetricCard,
    WithdrawalWizard,
    ReinvestmentPlanner,
    DistributionChart,
    DistributionSettings,
    WithdrawalHistoryTable,
    TaxReporting,
    OpportunityGrid,
    WithdrawalDetailModal,
    ReinvestmentModal,
    TaxReportModal,
    TimeRangeSelector,
    ExportButton,
  },
  setup() {
    // Composables
    const {
      totalProfits,
      availableForWithdrawal,
      monthlyProfits,
      profitTrend,
      withdrawalTrend,
      monthlyTrend,
      profitDistribution,
      distributionSettings,
      initiateWithdrawal,
      updateDistribution,
    } = useProfitManagement();

    const {
      withdrawalHistory,
      historyLoading,
      historyTimeRange,
      filteredWithdrawals,
      loadWithdrawalHistory,
    } = useWithdrawalHistory();

    const {
      taxableTransactions,
      currentFiscalYear,
      taxReport,
      generateTaxReport,
      exportTaxReport,
    } = useTaxReporting();

    const {
      reinvestmentOpportunities,
      selectedOpportunity,
      reinvestmentAmount,
      findReinvestmentOpportunities,
      executeReinvestment,
    } = useReinvestment();

    // Reactive state
    const distributionTimeRange = ref('1m');
    const showWithdrawalDetailModal = ref(false);
    const showReinvestmentModal = ref(false);
    const showTaxReportModal = ref(false);
    const selectedWithdrawal = ref<Withdrawal | null>(null);

    // Computed properties
    const historyRanges = computed(() => [
      { label: '1 Week', value: '1w' },
      { label: '1 Month', value: '1m' },
      { label: '3 Months', value: '3m' },
      { label: '1 Year', value: '1y' },
      { label: 'All Time', value: 'all' },
    ]);

    // Methods
    const handleDistributionSliceClick = (slice: ProfitDistribution) => {
      console.log('Distribution slice clicked:', slice);
      // Could show detailed breakdown for this category
    };

    const updateDistributionSettings = async (settings: any) => {
      await updateDistribution(settings);
    };

    const showWithdrawalDetail = (withdrawal: Withdrawal) => {
      selectedWithdrawal.value = withdrawal;
      showWithdrawalDetailModal.value = true;
    };

    const showReinvestmentPlan = (plan: any) => {
      selectedOpportunity.value = plan.opportunity;
      reinvestmentAmount.value = plan.amount;
      showReinvestmentModal.value = true;
    };

    const selectReinvestmentOpportunity = (opportunity: ReinvestmentOpportunity) => {
      selectedOpportunity.value = opportunity;
      reinvestmentAmount.value = opportunity.minAmount;
      showReinvestmentModal.value = true;
    };

    const confirmReinvestment = async () => {
      if (selectedOpportunity.value && reinvestmentAmount.value) {
        await executeReinvestment(selectedOpportunity.value, reinvestmentAmount.value);
        showReinvestmentModal.value = false;
      }
    };

    const generateTaxReportHandler = async (year: number) => {
      await generateTaxReport(year);
      showTaxReportModal.value = true;
    };

    const exportTaxReportHandler = async (format: string) => {
      await exportTaxReport(format);
    };

    const downloadTaxReport = () => {
      // Handle tax report download
      console.log('Download tax report:', taxReport.value);
    };

    // Lifecycle
    onMounted(() => {
      loadWithdrawalHistory();
      findReinvestmentOpportunities(availableForWithdrawal.value);
    });

    return {
      // State
      distributionTimeRange,
      showWithdrawalDetailModal,
      showReinvestmentModal,
      showTaxReportModal,
      selectedWithdrawal,
      
      // Computed
      totalProfits,
      availableForWithdrawal,
      monthlyProfits,
      profitTrend,
      withdrawalTrend,
      monthlyTrend,
      profitDistribution,
      distributionSettings,
      withdrawalHistory,
      historyLoading,
      historyTimeRange,
      filteredWithdrawals,
      taxableTransactions,
      currentFiscalYear,
      taxReport,
      reinvestmentOpportunities,
      selectedOpportunity,
      reinvestmentAmount,
      historyRanges,
      
      // Methods
      initiateWithdrawal,
      handleDistributionSliceClick,
      updateDistributionSettings,
      showWithdrawalDetail,
      showReinvestmentPlan,
      selectReinvestmentOpportunity,
      confirmReinvestment,
      generateTaxReport: generateTaxReportHandler,
      exportTaxReport: exportTaxReportHandler,
      downloadTaxReport,
    };
  },
});
</script>

<style scoped>
.profit-withdrawal {
  padding: 1rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.profit-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 2rem;
  flex-wrap: wrap;
}

.profit-summary {
  flex: 1;
  min-width: 300px;
}

.profit-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.profit-actions {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-width: 250px;
}

.distribution-section {
  background: var(--background-paper);
  border-radius: 8px;
  padding: 1.5rem;
}

.distribution-chart {
  height: 300px;
  margin-bottom: 1.5rem;
}

.distribution-settings {
  margin-top: 1.5rem;
}

.history-section {
  background: var(--background-paper);
  border-radius: 8px;
  padding: 1.5rem;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.history-controls {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.history-table {
  margin-top: 1rem;
}

.tax-section {
  background: var(--background-paper);
  border-radius: 8px;
  padding: 1.5rem;
}

.reinvestment-section {
  background: var(--background-paper);
  border-radius: 8px;
  padding: 1.5rem;
}

.opportunity-grid {
  margin-top: 1rem;
}

/* Responsive design */
@media (max-width: 1024px) {
  .profit-header {
    flex-direction: column;
  }
  
  .profit-actions {
    width: 100%;
    flex-direction: row;
    justify-content: space-between;
  }
}

@media (max-width: 768px) {
  .profit-metrics {
    grid-template-columns: 1fr;
  }
  
  .section-header {
    flex-direction: column;
    align-items: stretch;
    gap: 1rem;
  }
  
  .history-controls {
    justify-content: space-between;
  }
  
  .profit-actions {
    flex-direction: column;
  }
}

/* Metric card specific styling */
.total-profit {
  border-left: 4px solid var(--success-color);
}

.available-profit {
  border-left: 4px solid var(--primary-color);
}

.monthly-profit {
  border-left: 4px solid var(--warning-color);
}
</style>
