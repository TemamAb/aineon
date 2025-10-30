<template>
  <header class="dashboard-header">
    <div class="header-left">
      <div class="brand">
        <span class="title">AINEON MASTER DASHBOARD</span>
        <span class="subtitle">Integrated Trading System</span>
      </div>
    </div>

    <div class="header-center">
      <div class="profit-pulse">
        <span class="icon">Ì≤π</span>
        <span class="profit">${{ formatProfit(profit) }}</span>
        <span class="days">(14 days)</span>
      </div>
    </div>

    <div class="header-right">
      <div class="ai-controls">
        <button 
          class="mode-btn" 
          :class="{ active: aiMode === 'AUTONOMOUS' }"
          @click="setAIMode('AUTONOMOUS')"
        >
          <span class="icon">Ì¥ñ</span>
          <span>AUTONOMOUS</span>
        </button>
        <button 
          class="mode-btn" 
          :class="{ active: aiMode === 'MANUAL' }"
          @click="setAIMode('MANUAL')"
        >
          <span class="icon">Ì±®‚ÄçÌ≤º</span>
          <span>MANUAL</span>
        </button>
        <button 
          class="mode-btn" 
          :class="{ active: aiMode === 'COPILOT' }"
          @click="setAIMode('COPILOT')"
        >
          <span class="icon">Ì∑†</span>
          <span>COPILOT</span>
        </button>
      </div>

      <div class="control-group">
        <div class="dropdown">
          <button class="control-btn">
            <span class="icon">Ì¥Ñ</span>
            <span>{{ refreshInterval }}s</span>
            <span class="dropdown-arrow">‚ñº</span>
          </button>
          <div class="dropdown-content">
            <a @click="() => setRefreshInterval(1)">1s</a>
            <a @click="() => setRefreshInterval(2)">2s</a>
            <a @click="() => setRefreshInterval(5)">5s</a>
            <a @click="() => setRefreshInterval(10)">10s</a>
            <a @click="() => setRefreshInterval(30)">30s</a>
            <a @click="() => setRefreshInterval(0)">Manual</a>
          </div>
        </div>

        <div class="dropdown">
          <button class="control-btn">
            <span>{{ viewMode }}</span>
            <span class="dropdown-arrow">‚ñº</span>
          </button>
          <div class="dropdown-content">
            <a @click="() => setViewMode('DEFAULT')">Default View</a>
            <a @click="() => setViewMode('ADVANCED')">Advanced View</a>
          </div>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
import { ref } from 'vue';

interface Props {
  aiStatus: string;
  profit: number;
  refreshInterval: number;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: 'viewModeChange', mode: string): void;
  (e: 'refreshIntervalChange', interval: number): void;
  (e: 'aiModeChange', mode: string): void;
}>();

const aiMode = ref('AUTONOMOUS');
const viewMode = ref('DEFAULT');

const formatProfit = (amount: number) => {
  return amount.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
};

const setAIMode = (mode: string) => {
  aiMode.value = mode;
  emit('aiModeChange', mode);
  console.log('ÌæØ AI Mode changed to:', mode);
};

const setRefreshInterval = (interval: number) => {
  emit('refreshIntervalChange', interval);
  console.log('‚è∞ Refresh interval changed to:', interval, 'seconds');
};

const setViewMode = (mode: string) => {
  viewMode.value = mode;
  emit('viewModeChange', mode);
  console.log('Ì±Ä View mode changed to:', mode);
};
</script>

<style scoped>
.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background: #1a1a2e;
  border-bottom: 1px solid #2d2d44;
  color: white;
  height: 80px;
}

.header-left {
  flex: 1;
}

.brand {
  display: flex;
  flex-direction: column;
}

.brand .title {
  font-weight: 700;
  font-size: 1.4rem;
  color: #4caf50;
  margin-bottom: 0.1rem;
}

.brand .subtitle {
  font-size: 0.8rem;
  color: #ccc;
  opacity: 0.8;
}

.header-center {
  flex: 1;
  display: flex;
  justify-content: center;
}

.profit-pulse {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1.4rem;
  font-weight: 700;
  background: linear-gradient(135deg, #4caf50, #45a049);
  padding: 0.8rem 1.5rem;
  border-radius: 25px;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
}

.profit {
  font-size: 1.5rem;
}

.days {
  font-size: 0.9rem;
  opacity: 0.8;
}

.header-right {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 1.5rem;
}

.ai-controls {
  display: flex;
  background: #2d2d44;
  border-radius: 10px;
  padding: 4px;
  gap: 2px;
}

.mode-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 1rem;
  border: none;
  background: transparent;
  color: #ccc;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
  transition: all 0.3s ease;
}

.mode-btn.active {
  background: #4caf50;
  color: white;
}

.mode-btn:hover:not(.active) {
  background: #393954;
  color: white;
}

.control-group {
  display: flex;
  align-items: center;
  gap: 0.8rem;
}

.control-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 1rem;
  background: #2d2d44;
  border: 1px solid #393954;
  border-radius: 8px;
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.9rem;
  font-weight: 500;
}

.control-btn:hover {
  background: #393954;
  border-color: #4caf50;
}

.dropdown {
  position: relative;
  display: inline-block;
}

.dropdown-content {
  display: none;
  position: absolute;
  background: #2d2d44;
  min-width: 120px;
  box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
  z-index: 1000;
  border-radius: 8px;
  overflow: hidden;
  right: 0;
}

.dropdown-content a {
  color: white;
  padding: 12px 16px;
  text-decoration: none;
  display: block;
  cursor: pointer;
  transition: background 0.3s ease;
  font-size: 0.9rem;
}

.dropdown-content a:hover {
  background: #393954;
}

.dropdown:hover .dropdown-content {
  display: block;
}

.dropdown-arrow {
  font-size: 0.7rem;
  margin-left: 0.3rem;
}
</style>
