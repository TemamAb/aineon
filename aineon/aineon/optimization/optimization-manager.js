class OptimizationManager {
  constructor() {
    this.optimizationEngine = new (require('./continuous-optimization-engine'))();
    this.isRunning = false;
    this.startTime = null;
  }

  async startOptimizationSystem() {
    if (this.isRunning) {
      console.log('‚ö†Ô∏è  Optimization system is already running');
      return false;
    }

    console.log('ÌæØ STARTING AINEON CONTINUOUS OPTIMIZATION SYSTEM');
    console.log('=================================================');
    
    this.startTime = Date.now();
    this.isRunning = true;
    
    try {
      await this.optimizationEngine.startContinuousOptimization();
      console.log('‚úÖ Continuous optimization system started successfully');
      
      // Start periodic reporting
      this.startPeriodicReporting();
      
      return true;
    } catch (error) {
      console.error('‚ùå Failed to start optimization system:', error);
      this.isRunning = false;
      return false;
    }
  }

  stopOptimizationSystem() {
    if (!this.isRunning) {
      console.log('‚ö†Ô∏è  Optimization system is not running');
      return false;
    }

    console.log('Ìªë STOPPING CONTINUOUS OPTIMIZATION SYSTEM');
    this.isRunning = false;
    
    // In a real implementation, we would properly shut down intervals
    console.log('‚úÖ Optimization system stopped');
    return true;
  }

  startPeriodicReporting() {
    // Hourly performance reports
    setInterval(() => {
      if (this.isRunning) {
        this.generateHourlyReport();
      }
    }, 3600000); // 1 hour
    
    // Daily comprehensive reports
    setInterval(() => {
      if (this.isRunning) {
        this.generateDailyReport();
      }
    }, 86400000); // 24 hours
  }

  generateHourlyReport() {
    const stats = this.optimizationEngine.getOptimizationStatistics();
    const deltaReport = this.optimizationEngine.deltaTracker.getPerformanceReport();
    
    console.log('\n‚è∞ HOURLY OPTIMIZATION REPORT');
    console.log('===========================');
    console.log(`Cycles Completed: ${stats.cycleCount}`);
    console.log(`Total Improvement: ${(deltaReport.totalImprovement * 100).toFixed(3)}%`);
    console.log(`Compounding Effect: ${deltaReport.compoundingMultiplier.toFixed(4)}x`);
    console.log(`Projected Daily Profit: $${deltaReport.dailyProjection.projectedPerformance.toFixed(0)}`);
    console.log(`System Uptime: ${this.formatUptime()}`);
  }

  generateDailyReport() {
    const deltaReport = this.optimizationEngine.deltaTracker.getPerformanceReport();
    
    console.log('\nÌ≥Ö DAILY OPTIMIZATION REPORT');
    console.log('==========================');
    console.log(`Total Cycles: ${deltaReport.totalCycles}`);
    console.log(`Total Accumulated Improvement: ${(deltaReport.totalImprovement * 100).toFixed(3)}%`);
    console.log(`Compounding Multiplier: ${deltaReport.compoundingMultiplier.toFixed(4)}x`);
    console.log(`Baseline Daily Profit: $${deltaReport.currentBaseline.toFixed(0)}`);
    console.log(`Optimized Daily Profit: $${deltaReport.dailyProjection.projectedPerformance.toFixed(0)}`);
    console.log(`Improvement: +${((deltaReport.dailyProjection.projectedPerformance / deltaReport.currentBaseline - 1) * 100).toFixed(2)}%`);
    
    console.log('\nÌæØ DIMENSION PERFORMANCE:');
    deltaReport.optimalDimensions.forEach((dim, index) => {
      console.log(`   ${index + 1}. ${dim.dimension}: +${(dim.averageImprovement * 100).toFixed(3)}% (${(dim.contribution * 100).toFixed(1)}% of total)`);
    });
  }

  formatUptime() {
    if (!this.startTime) return '0s';
    
    const uptime = Date.now() - this.startTime;
    const hours = Math.floor(uptime / 3600000);
    const minutes = Math.floor((uptime % 3600000) / 60000);
    const seconds = Math.floor((uptime % 60000) / 1000);
    
    return `${hours}h ${minutes}m ${seconds}s`;
  }

  getSystemStatus() {
    return {
      isRunning: this.isRunning,
      startTime: this.startTime,
      uptime: this.formatUptime(),
      optimizationStats: this.isRunning ? this.optimizationEngine.getOptimizationStatistics() : null,
      deltaReport: this.isRunning ? this.optimizationEngine.deltaTracker.getPerformanceReport() : null
    };
  }

  async triggerManualOptimizationCycle() {
    if (!this.isRunning) {
      console.log('‚ö†Ô∏è  Optimization system is not running');
      return false;
    }

    console.log('Ì¥ß MANUAL OPTIMIZATION CYCLE TRIGGERED');
    await this.optimizationEngine.optimizationCycle();
    return true;
  }
}

module.exports = OptimizationManager;
