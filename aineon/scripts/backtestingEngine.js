// Historical Strategy Performance Simulation
const runBacktest = (strategy, period = '30d') => {
  const historicalResults = {
    totalProfit: 0,
    winningTrades: 0,
    losingTrades: 0,
    maxDrawdown: 0,
    sharpeRatio: 0,
    totalTrades: 0
  };

  // Simulate backtesting results based on strategy
  if (strategy.riskLevel === "LOW") {
    historicalResults.totalProfit = 450000;
    historicalResults.winningTrades = 85;
    historicalResults.losingTrades = 15; 
    historicalResults.maxDrawdown = -0.05;
    historicalResults.sharpeRatio = 2.1;
    historicalResults.totalTrades = 100;
  } else if (strategy.riskLevel === "MEDIUM") {
    historicalResults.totalProfit = 1200000;
    historicalResults.winningTrades = 70;
    historicalResults.losingTrades = 30;
    historicalResults.maxDrawdown = -0.12;
    historicalResults.sharpeRatio = 1.8;
    historicalResults.totalTrades = 150;
  } else { // HIGH
    historicalResults.totalProfit = 3500000;
    historicalResults.winningTrades = 60;
    historicalResults.losingTrades = 40;
    historicalResults.maxDrawdown = -0.25;
    historicalResults.sharpeRatio = 1.5;
    historicalResults.totalTrades = 200;
  }

  return {
    ...historicalResults,
    winRate: (historicalResults.winningTrades / historicalResults.totalTrades) * 100,
    avgProfitPerTrade: historicalResults.totalProfit / historicalResults.totalTrades,
    backtestPeriod: period,
    timestamp: new Date().toISOString()
  };
};

return runBacktest;
