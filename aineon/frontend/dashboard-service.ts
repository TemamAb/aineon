import { ref } from 'vue';

export interface DashboardMetrics {
  system: any;
  wallet: any;
  trading: any;
  optimization: any;
  monitoring: any;
  ai: any;
  profit: any;
  flashloan: any;
  security: any;
  health: any;
}

export interface AIConfig {
  mode: 'AUTONOMOUS' | 'SEMI_AUTO' | 'MANUAL';
  confidence: number;
  optimizationLevel: number;
}

class DashboardService {
  private baseUrl = 'http://localhost:5000/api'; // Port 5000
  private currentMetrics = ref<DashboardMetrics | null>(null);
  private aiConfig = ref<AIConfig>({
    mode: 'AUTONOMOUS',
    confidence: 0.95,
    optimizationLevel: 100
  });

  async getMetrics(): Promise<DashboardMetrics> {
    try {
      const response = await fetch(`${this.baseUrl}/metrics`);
      const data = await response.json();
      this.currentMetrics.value = data;
      return data;
    } catch (error) {
      console.error('Failed to fetch dashboard metrics:', error);
      throw error;
    }
  }

  async enableFullAutonomy(): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/ai/autonomy/enable`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'AUTONOMOUS' })
      });
      
      if (response.ok) {
        this.aiConfig.value.mode = 'AUTONOMOUS';
      }
    } catch (error) {
      console.error('Failed to enable autonomy:', error);
      throw error;
    }
  }

  async optimizeParameters(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/ai/optimize/parameters`, {
      method: 'POST'
    });
    return response.json();
  }

  async deployLive(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/deployment/live`, {
      method: 'POST'
    });
    return response.json();
  }

  getCurrentMetrics() {
    return this.currentMetrics;
  }

  getAIConfig() {
    return this.aiConfig;
  }
}

export const dashboardService = new DashboardService();
