class MultiPairArbitrage {
  constructor() {
    this.maxPairsPerChain = 50;
    this.strategyTypes = {
      triangular: { weight: 0.25, maxAllocation: 0.15 },
      composite: { weight: 0.20, maxAllocation: 0.12 },
      volatility: { weight: 0.18, maxAllocation: 0.10 },
      correlation: { weight: 0.22, maxAllocation: 0.14 },
      crossAsset: { weight: 0.15, maxAllocation: 0.08 }
    };
  }

  async discoverArbitrageUniverse(chainId, availableCapital) {
    const pairs = await this.scanTradingPairs(chainId);
    const opportunities = await Promise.all(
      pairs.map(pair => this.analyzeMultiStrategyOpportunities(pair, availableCapital))
    );

    return this.optimizePortfolioAllocation(opportunities.flat(), availableCapital);
  }

  async analyzeMultiStrategyOpportunities(pair, capital) {
    const strategies = await Promise.all([
      this.analyzeTriangularArbitrage(pair, capital * 0.15),
      this.analyzeCompositeArbitrage(pair, capital * 0.12),
      this.analyzeVolatilityArbitrage(pair, capital * 0.10),
      this.analyzeCorrelationArbitrage(pair, capital * 0.14),
      this.analyzeCrossAssetArbitrage(pair, capital * 0.08)
    ]);

    return strategies.filter(opp => opp.expectedProfit > 0.005 && opp.confidence > 0.8);
  }

  async analyzeTriangularArbitrage(pair, allocation) {
    // ETH → BTC → USDC → ETH arbitrage
    const paths = await this.findTriangularPaths(pair);
    const profitablePaths = paths.filter(path => path.expectedProfit > 0.008);
    
    return {
      type: 'triangular',
      pair: pair.symbol,
      expectedProfit: profitablePaths.reduce((max, path) => Math.max(max, path.expectedProfit), 0),
      confidence: 0.85,
      requiredCapital: allocation,
      paths: profitablePaths.slice(0, 3)
    };
  }

  async analyzeCompositeArbitrage(pair, allocation) {
    // Liquidity provision + arbitrage combination
    const [arbitrageProfit, yieldProfit] = await Promise.all([
      this.calculateArbitrageProfit(pair),
      this.calculateYieldOpportunity(pair)
    ]);

    const compositeProfit = arbitrageProfit + (yieldProfit * 0.6); // Risk-adjusted
    return {
      type: 'composite',
      pair: pair.symbol,
      expectedProfit: compositeProfit,
      confidence: 0.82,
      requiredCapital: allocation,
      components: { arbitrage: arbitrageProfit, yield: yieldProfit }
    };
  }

  optimizePortfolioAllocation(opportunities, totalCapital) {
    const sorted = opportunities.sort((a, b) => b.expectedProfit - a.expectedProfit);
    const allocation = {};
    let remainingCapital = totalCapital;

    for (const opp of sorted) {
      if (remainingCapital <= 0) break;
      
      const maxAllocation = this.strategyTypes[opp.type].maxAllocation * totalCapital;
      const allocated = Math.min(opp.requiredCapital, maxAllocation, remainingCapital);
      
      if (allocated >= opp.requiredCapital * 0.5) { // Minimum 50% of required
        allocation[opp.pair + '_' + opp.type] = {
          ...opp,
          allocatedCapital: allocated,
          efficiency: allocated / opp.requiredCapital
        };
        remainingCapital -= allocated;
      }
    }

    return {
      allocation,
      totalAllocated: totalCapital - remainingCapital,
      utilization: (totalCapital - remainingCapital) / totalCapital,
      opportunityCount: Object.keys(allocation).length
    };
  }
}
module.exports = MultiPairArbitrage;
