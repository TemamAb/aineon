// Pre-configured Strategy Templates
const templates = {
  CONSERVATIVE: {
    name: "í¿¢ Conservative Arbitrage",
    description: "Low-risk cross-chain arbitrage with minimal leverage",
    config: {
      riskLevel: "LOW",
      maxPositionSize: 100000,
      leverage: 1,
      stopLoss: -0.02,
      chains: ["ETHEREUM", "BSC"],
      minConfidence: 0.85
    },
    allocation: { flashLoan: 40, crossChain: 30, mevBots: 20, reserve: 10 }
  },
  
  BALANCED: {
    name: "í¿¡ Balanced Multi-Strategy", 
    description: "Mixed strategy with moderate risk across all opportunities",
    config: {
      riskLevel: "MEDIUM", 
      maxPositionSize: 500000,
      leverage: 3,
      stopLoss: -0.05,
      chains: ["ETHEREUM", "BSC", "POLYGON", "ARBITRUM"],
      minConfidence: 0.75
    },
    allocation: { flashLoan: 50, crossChain: 25, mevBots: 15, reserve: 10 }
  },
  
  AGGRESSIVE: {
    name: "í´´ Aggressive MEV Hunting",
    description: "High-frequency MEV extraction with maximum leverage",
    config: {
      riskLevel: "HIGH",
      maxPositionSize: 2000000, 
      leverage: 10,
      stopLoss: -0.10,
      chains: ["ETHEREUM", "BSC", "POLYGON", "ARBITRUM", "OPTIMISM"],
      minConfidence: 0.60
    },
    allocation: { flashLoan: 60, crossChain: 20, mevBots: 15, reserve: 5 }
  }
};

return templates;
