<!-- PLATINUM SOURCES: Vue Chart.js, ApexCharts -->
<!-- CONTINUAL LEARNING: Display preference learning, chart optimization -->

<template>
  <div class="live-monitoring">
    <!-- Header with Key Metrics -->
    <div class="monitoring-header">
      <div class="metrics-overview">
        <metric-card
          v-for="metric in realTimeMetrics"
          :key="metric.id"
          :title="metric.title"
          :value="metric.value"
          :trend="metric.trend"
          :loading="metric.loading"
          class="metric-item"
        />
      </div>
      
      <div class="controls">
        <time-range-selector v-model="timeRange" />
        <refresh-controls 
          :auto-refresh="autoRefresh"
          @auto-refresh-change="handleAutoRefreshChange"
        />
        <chart-style-selector 
          v-model="chartStyle"
          @style-change="handleChartStyleChange"
        />
      </div>
    </div>

    <!-- Main Chart Grid -->
    <div class="chart-grid" :class="`chart-style-${chartStyle}`">
      <div class="chart-row">
        <real-time-chart
          :data="priceChartData"
          :options="priceChartOptions"
          title="Price Movement"
          chart-type="line"
          class="chart-item main-chart"
          @chart-interaction="handleChartInteraction"
        />
        
        <real-time-chart
          :data="volumeChartData"
          :options="volumeChartOptions"
          title="Trading Volume"
          chart-type="bar"
          class="chart-item secondary-chart"
        />
      </div>
      
      <div class="chart-row">
        <performance-gauge
          :value="performanceScore"
          :thresholds="performanceThresholds"
          title="System Performance"
          class="gauge-item"
        />
        
        <heat-map-chart
          :data="arbitrageHeatmapData"
          :options="heatmapOptions"
          title="Arbitrage Opportunities Heatmap"
          class="heatmap-item"
        />
      </div>
    </div>

    <!-- Real-time Activity Feed -->
    <div class="activity-section">
      <h3>Real-time Activity</h3>
      <activity-feed
        :activities="recentActivities"
        :loading="activitiesLoading"
        @activity-click="handleActivityClick"
        class="activity-feed"
      />
    </div>

    <!-- Alert Panel -->
    <alert-panel
      :alerts="activeAlerts"
      :severity-filter="alertSeverityFilter"
      @alert-action="handleAlertAction"
      class="alert-panel"
    />
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, onMounted, onUnmounted } from 'vue';
import { useWebSocket } from '../composables/useWebSocket';
import { useChartPreferences } from '../composables/useChartPreferences';
import { useRealTimeData } from '../composables/useRealTimeData';

// Components
import MetricCard from '../components/Metrics/MetricCard.vue';
import RealTimeChart from '../components/Charts/RealTimeChart.vue';
import PerformanceGauge from '../components/Charts/PerformanceGauge.vue';
import HeatMapChart from '../components/Charts/HeatMapChart.vue';
import ActivityFeed from '../components/Activity/ActivityFeed.vue';
import AlertPanel from '../components/Alerts/AlertPanel.vue';
import TimeRangeSelector from '../components/Controls/TimeRangeSelector.vue';
import RefreshControls from '../components/Controls/RefreshControls.vue';
import ChartStyleSelector from '../components/Controls/ChartStyleSelector.vue';

// Types
interface RealTimeMetric {
  id: string;
  title: string;
  value: number | string;
  trend: 'up' | 'down' | 'stable';
  loading: boolean;
}

export default defineComponent({
  name: 'LiveMonitoring',
  components: {
    MetricCard,
    RealTimeChart,
    PerformanceGauge,
    HeatMapChart,
    ActivityFeed,
    AlertPanel,
    TimeRangeSelector,
    RefreshControls,
    ChartStyleSelector,
  },
  setup() {
    // Composables
    const { connected, lastMessage, sendMessage } = useWebSocket();
    const { chartPreferences, updateChartPreference } = useChartPreferences();
    const { 
      realTimeData, 
      metrics, 
      activities, 
      alerts,
      subscribe,
      unsubscribe 
    } = useRealTimeData();

    // Reactive state
    const timeRange = ref('1h');
    const autoRefresh = ref(true);
    const chartStyle = ref(chartPreferences.value.style || 'standard');
    const activitiesLoading = ref(false);

    // Computed properties
    const realTimeMetrics = computed<RealTimeMetric[]>(() => [
      {
        id: 'total-trades',
        title: 'Total Trades',
        value: metrics.value.totalTrades,
        trend: metrics.value.tradeTrend,
        loading: !connected.value,
      },
      {
        id: 'success-rate',
        title: 'Success Rate',
        value: `${metrics.value.successRate}%`,
        trend: metrics.value.successTrend,
        loading: !connected.value,
      },
      {
        id: 'total-profit',
        title: 'Total Profit',
        value: `$${metrics.value.totalProfit.toLocaleString()}`,
        trend: metrics.value.profitTrend,
        loading: !connected.value,
      },
      {
        id: 'active-strategies',
        title: 'Active Strategies',
        value: metrics.value.activeStrategies,
        trend: 'stable',
        loading: !connected.value,
      },
    ]);

    const priceChartData = computed(() => 
      realTimeData.value.priceData.slice(-100) // Last 100 data points
    );

    const volumeChartData = computed(() => 
      realTimeData.value.volumeData.slice(-50) // Last 50 data points
    );

    const arbitrageHeatmapData = computed(() => 
      realTimeData.value.arbitrageOpportunities
    );

    const recentActivities = computed(() => 
      activities.value.slice(0, 20) // Last 20 activities
    );

    const activeAlerts = computed(() => 
      alerts.value.filter(alert => alert.active)
    );

    // Chart configurations
    const priceChartOptions = computed(() => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: chartPreferences.value.animationEnabled ? 1000 : 0,
      },
      plugins: {
        legend: {
          display: chartPreferences.value.showLegends,
        },
      },
    }));

    const performanceThresholds = ref([
      { value: 0, color: '#ef4444' },
      { value: 70, color: '#f59e0b' },
      { value: 90, color: '#10b981' },
    ]);

    // Methods
    const handleAutoRefreshChange = (enabled: boolean) => {
      autoRefresh.value = enabled;
      // Learn user preference for auto-refresh
      updateChartPreference('autoRefresh', enabled);
    };

    const handleChartStyleChange = (style: string) => {
      chartStyle.value = style;
      // Learn user chart style preference
      updateChartPreference('style', style);
    };

    const handleChartInteraction = (interactionData: any) => {
      // Track user interaction for behavior learning
      updateChartPreference('lastInteraction', {
        type: interactionData.type,
        timestamp: Date.now(),
        chartType: interactionData.chartType,
      });
    };

    const handleActivityClick = (activity: any) => {
      // Handle activity item click
      console.log('Activity clicked:', activity);
    };

    const handleAlertAction = (alert: any, action: string) => {
      // Handle alert actions
      sendMessage('alert_action', { alertId: alert.id, action });
    };

    // Lifecycle
    onMounted(() => {
      subscribe('trading_metrics');
      subscribe('price_data');
      subscribe('activity_feed');
      subscribe('alerts');
    });

    onUnmounted(() => {
      unsubscribe('trading_metrics');
      unsubscribe('price_data');
      unsubscribe('activity_feed');
      unsubscribe('alerts');
    });

    return {
      // State
      timeRange,
      autoRefresh,
      chartStyle,
      activitiesLoading,
      
      // Computed
      realTimeMetrics,
      priceChartData,
      volumeChartData,
      arbitrageHeatmapData,
      recentActivities,
      activeAlerts,
      performanceScore: computed(() => metrics.value.performanceScore),
      
      // Configurations
      priceChartOptions,
      volumeChartOptions: priceChartOptions, // Reuse same options
      heatmapOptions: { responsive: true },
      performanceThresholds,
      alertSeverityFilter: ref(['critical', 'high']),
      
      // Methods
      handleAutoRefreshChange,
      handleChartStyleChange,
      handleChartInteraction,
      handleActivityClick,
      handleAlertAction,
    };
  },
});
</script>

<style scoped>
.live-monitoring {
  padding: 1rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.monitoring-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  flex-wrap: wrap;
}

.metrics-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  flex: 1;
}

.controls {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
}

.chart-grid {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  flex: 1;
  min-height: 500px;
}

.chart-row {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 1rem;
  height: 300px;
}

.chart-item, .gauge-item, .heatmap-item {
  background: var(--background-paper);
  border-radius: 8px;
  padding: 1rem;
}

.activity-section, .alert-panel {
  background: var(--background-paper);
  border-radius: 8px;
  padding: 1rem;
}

/* Responsive design */
@media (max-width: 768px) {
  .monitoring-header {
    flex-direction: column;
  }
  
  .chart-row {
    grid-template-columns: 1fr;
    height: auto;
  }
  
  .metrics-overview {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 480px) {
  .metrics-overview {
    grid-template-columns: 1fr;
  }
  
  .controls {
    flex-direction: column;
    align-items: stretch;
  }
}

/* Chart style variants */
.chart-style-minimal .chart-item {
  background: transparent;
  border: 1px solid var(--border-color);
}

.chart-style-detailed .chart-item {
  background: var(--background-paper);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
</style>
