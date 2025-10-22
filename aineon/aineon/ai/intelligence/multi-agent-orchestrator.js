// AI AGENT COORDINATION & CONSENSUS
class MultiAgentOrchestrator {
  constructor() {
    this.agents = {
      strategic: new (require('../command/captain-strategic'))(),
      tactical: new (require('../swarm/opportunity-cluster'))(),
      risk: new (require('../command/risk-sovereign'))(),
      execution: new (require('../execution/smart-relayer'))()
    };
    this.consensusThreshold = 0.75; // 75% agreement required
  }

  async makeTradingDecision(marketData, portfolioState) {
    const agentOpinions = await this.collectAgentOpinions(marketData, portfolioState);
    const consensus = await this.calculateConsensus(agentOpinions);
    
    if (consensus.score >= this.consensusThreshold) {
      return await this.executeConsensusDecision(consensus, portfolioState);
    } else {
      return await this.handleDisagreement(agentOpinions, portfolioState);
    }
  }

  async collectAgentOpinions(marketData, portfolioState) {
    const opinions = {};
    
    opinions.strategic = await this.agents.strategic.analyzeMarketRegime(marketData);
    opinions.tactical = await this.agents.tactical.findOpportunities(marketData);
    opinions.risk = await this.agents.risk.assessPortfolioRisk(portfolioState);
    opinions.execution = await this.agents.execution.assessFeasibility(marketData);
    
    return opinions;
  }

  async calculateConsensus(opinions) {
    const signals = Object.values(opinions).map(opinion => opinion.signal);
    const confidences = Object.values(opinions).map(opinion => opinion.confidence);
    
    const averageConfidence = confidences.reduce((a, b) => a + b, 0) / confidences.length;
    const agreement = this.calculateAgreement(signals);
    
    return {
      decision: this.resolveSignals(signals),
      score: (averageConfidence + agreement) / 2,
      details: opinions,
      timestamp: Date.now()
    };
  }

  calculateAgreement(signals) {
    const signalCounts = {};
    signals.forEach(signal => {
      signalCounts[signal] = (signalCounts[signal] || 0) + 1;
    });
    
    const maxCount = Math.max(...Object.values(signalCounts));
    return maxCount / signals.length;
  }

  resolveSignals(signals) {
    const counts = {};
    signals.forEach(signal => {
      counts[signal] = (counts[signal] || 0) + 1;
    });
    
    return Object.keys(counts).reduce((a, b) => 
      counts[a] > counts[b] ? a : b
    );
  }

  async executeConsensusDecision(consensus, portfolioState) {
    const executionPlan = await this.createExecutionPlan(consensus, portfolioState);
    const riskCheck = await this.finalRiskAssessment(executionPlan);
    
    if (riskCheck.approved) {
      return await this.agents.execution.executePlan(executionPlan);
    } else {
      return { status: 'BLOCKED', reason: riskCheck.reason, plan: executionPlan };
    }
  }
}
module.exports = MultiAgentOrchestrator;
