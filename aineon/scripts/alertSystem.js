// Real-time Multi-level Alert System
const alerts = [];
const currentPerformance = getTradingPerformanceQuery.data;
const currentStatus = getModulesStatusQuery.data;

// AI Confidence Alerts
if (currentPerformance.ai_confidence < 0.7) {
  alerts.push({
    level: "CRITICAL",
    type: "AI_CONFIDENCE",
    message: `AI Confidence dropped to ${(currentPerformance.ai_confidence * 100).toFixed(1)}%`,
    timestamp: new Date().toISOString(),
    action: "PAUSE_TRADING"
  });
}

// Profit/Loss Alerts
const hourlyProfit = currentPerformance.profit_last_hour;
if (hourlyProfit < -5000) {
  alerts.push({
    level: "HIGH",
    type: "LOSS_THRESHOLD", 
    message: `Significant loss detected: $${Math.abs(hourlyProfit)} last hour`,
    timestamp: new Date().toISOString(),
    action: "REVIEW_STRATEGY"
  });
}

// System Health Alerts
if (currentStatus.performance.health < 0.9) {
  alerts.push({
    level: "MEDIUM",
    type: "SYSTEM_HEALTH",
    message: "System health degraded - performance impact possible",
    timestamp: new Date().toISOString(),
    action: "INVESTIGATE"
  });
}

// Security Threat Alerts
if (currentStatus.security.threats > 0) {
  alerts.push({
    level: "CRITICAL", 
    type: "SECURITY_THREAT",
    message: `${currentStatus.security.threats} security threats detected`,
    timestamp: new Date().toISOString(),
    action: "EMERGENCY_REVIEW"
  });
}

return {
  activeAlerts: alerts,
  totalCount: alerts.length,
  criticalCount: alerts.filter(a => a.level === "CRITICAL").length,
  lastUpdated: new Date().toISOString()
};
