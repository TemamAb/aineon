// Comprehensive Security Event Logging
const securityEvents = [];

const logSecurityEvent = (event) => {
  const securityEvent = {
    id: generateId(),
    timestamp: new Date().toISOString(),
    level: event.level || "INFO",
    type: event.type,
    message: event.message,
    user: event.user || "SYSTEM",
    action: event.action,
    metadata: event.metadata || {}
  };

  securityEvents.unshift(securityEvent); // Add to beginning for reverse chronological
  
  // Keep only last 1000 events
  if (securityEvents.length > 1000) {
    securityEvents.pop();
  }

  return securityEvent;
};

// Log initial security state
logSecurityEvent({
  level: "INFO",
  type: "SYSTEM_STARTUP",
  message: "AINEON Security Monitoring Activated",
  action: "MONITOR"
});

return {
  logSecurityEvent,
  getSecurityEvents: () => securityEvents.slice(0, 50), // Return recent 50
  getEventCount: () => securityEvents.length,
  clearEvents: () => securityEvents.length = 0
};
