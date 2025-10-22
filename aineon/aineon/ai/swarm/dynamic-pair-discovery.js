class DynamicPairDiscovery {
  constructor() {
    this.pairCache = new Map();
    this.volumeThreshold = 1000000; // $1M daily volume
  }

  async discoverViablePairs(chainId, count = 50) {
    const allPairs = await this.fetchChainPairs(chainId);
    const scoredPairs = await Promise.all(
      allPairs.map(pair => this.scorePairViability(pair))
    );

    return scoredPairs
      .filter(pair => pair.score > 0.7)
      .sort((a, b) => b.score - a.score)
      .slice(0, count)
      .map(pair => pair.data);
  }

  async scorePairViability(pair) {
    const [liquidityScore, volatilityScore, correlationScore] = await Promise.all([
      this.calculateLiquidityScore(pair),
      this.calculateVolatilityScore(pair),
      this.calculateCorrelationScore(pair)
    ]);

    const compositeScore = (liquidityScore * 0.4) + (volatilityScore * 0.35) + (correlationScore * 0.25);
    
    return {
      data: pair,
      score: compositeScore,
      components: { liquidityScore, volatilityScore, correlationScore }
    };
  }

  async calculateLiquidityScore(pair) {
    const volume24h = pair.volumeUSD || 0;
    const liquidity = pair.liquidityUSD || 0;
    
    const volumeScore = Math.min(volume24h / this.volumeThreshold, 1);
    const liquidityScore = Math.min(liquidity / 5000000, 1); // $5M liquidity cap
    
    return (volumeScore * 0.6) + (liquidityScore * 0.4);
  }

  async calculateVolatilityScore(pair) {
    // Optimal volatility for arbitrage: not too low (no opportunities), not too high (too risky)
    const volatility = pair.volatility24h || 0.02;
    const idealVolatility = 0.15; // 15% daily volatility ideal
    
    const distance = Math.abs(volatility - idealVolatility);
    return Math.max(0, 1 - (distance / 0.3)); // Score drops beyond 30% distance
  }

  async calculateCorrelationScore(pair) {
    // Pairs with medium correlation are best for diversified arbitrage
    const correlation = await this.getPairCorrelation(pair);
    const idealCorrelation = 0.4;
    
    const distance = Math.abs(correlation - idealCorrelation);
    return Math.max(0, 1 - (distance / 0.6)); // Score drops beyond 0.6 distance
  }

  async findCorrelatedPairs(basePair, allPairs, threshold = 0.3) {
    const correlations = await Promise.all(
      allPairs.map(targetPair => 
        this.calculatePairCorrelation(basePair, targetPair)
      )
    );

    return allPairs.filter((_, index) => correlations[index] > threshold);
  }

  async getOptimalPairCluster(pairs, clusterSize = 8) {
    // Find groups of pairs with optimal inter-correlation for diversified arbitrage
    const clusters = [];
    
    for (const basePair of pairs.slice(0, 20)) { // Check top 20 pairs as cluster seeds
      const correlated = await this.findCorrelatedPairs(basePair, pairs, 0.2);
      const uncorrelated = pairs.filter(p => !correlated.includes(p));
      
      if (correlated.length >= 3 && uncorrelated.length >= 5) {
        clusters.push({
          base: basePair,
          correlated: correlated.slice(0, 3),
          uncorrelated: uncorrelated.slice(0, 5),
          diversityScore: this.calculateClusterDiversity(correlated, uncorrelated)
        });
      }
    }

    return clusters.sort((a, b) => b.diversityScore - a.diversityScore)[0];
  }
}
module.exports = DynamicPairDiscovery;
