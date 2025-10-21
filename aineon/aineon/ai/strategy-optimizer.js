// strategy-optimizer.js - Continuous Parameter Optimization & Learning
// Reverse-engineered from DEAP Python, Optuna hyperparameter optimization

const tf = require('@tensorflow/tfjs');
const { GeneticOptimizer } = require('./genetic-optimizer');

class StrategyOptimizer {
    constructor(config) {
        this.config = config;
        this.geneticOptimizer = new GeneticOptimizer();
        this.performanceHistory = [];
        this.parameterEvolution = [];
        this.adaptationRate = config.adaptationRate || 0.1;
    }

    async initialize() {
        console.log('í´„ Initializing Strategy Optimizer with Continual Learning...');
        await this.geneticOptimizer.initialize();
        this.loadHistoricalPerformance();
        console.log('âœ… Strategy Optimizer ready');
    }

    async optimizeStrategy(strategy, marketRegime, historicalData) {
        // Get base parameters
        const baseParams = strategy.parameters;
        
        // Apply continual learning adjustments
        const learnedAdjustments = await this.getLearnedAdjustments(strategy, marketRegime);
        
        // Genetic optimization for fine-tuning
        const optimizedParams = await this.geneticOptimizer.optimize(
            baseParams,
            learnedAdjustments,
            marketRegime
        );

        // Validate parameters
        const validatedParams = this.validateParameters(optimizedParams, strategy);
        
        return {
            ...validatedParams,
            confidence: await this.calculateParameterConfidence(strategy, validatedParams),
            learningSource: this.getLearningSource(learnedAdjustments),
            adaptationLevel: this.calculateAdaptationLevel(learnedAdjustments)
        };
    }

    async getLearnedAdjustments(strategy, marketRegime) {
        // Get adjustments from continual learning
        const adjustments = {
            riskMultiplier: await this.learnRiskMultiplier(strategy, marketRegime),
            positionSize: await this.learnPositionSizing(strategy, marketRegime),
            stopLoss: await this.learnStopLoss(strategy, marketRegime),
            takeProfit: await this.learnTakeProfit(strategy, marketRegime),
            timing: await this.learnExecutionTiming(strategy, marketRegime)
        };

        return this.applyForgettingMechanism(adjustments, strategy.lastUpdate);
    }

    async learnRiskMultiplier(strategy, marketRegime) {
        // Learn optimal risk multiplier from historical performance
        const regimePerformance = this.getRegimePerformance(strategy, marketRegime);
        
        if (regimePerformance.length < 10) {
            return 1.0; // Default multiplier
        }

        // Calculate performance-weighted adjustment
        const performanceWeights = regimePerformance.map(p => p.profit / p.maxDrawdown);
        const avgPerformance = performanceWeights.reduce((a, b) => a + b) / performanceWeights.length;
        
        // Adjust risk based on performance (higher performance = slightly more risk)
        return Math.min(2.0, Math.max(0.5, 1.0 + (avgPerformance * 0.1)));
    }

    async learnPositionSizing(strategy, marketRegime) {
        // Adaptive position sizing based on volatility and performance
        const recentPerformance = this.getRecentPerformance(strategy, 50); // Last 50 trades
        const volatility = await this.getMarketVolatility(marketRegime);
        
        if (recentPerformance.length === 0) return 1.0;

        const winRate = recentPerformance.filter(p => p.profit > 0).length / recentPerformance.length;
        const avgProfit = recentPerformance.reduce((sum, p) => sum + p.profit, 0) / recentPerformance.length;
        
        // Increase position size with higher win rate and profit, decrease with higher volatility
        const sizeAdjustment = (winRate * 0.5) + (avgProfit * 2) - (volatility * 0.3);
        
        return Math.min(1.5, Math.max(0.3, 1.0 + sizeAdjustment));
    }

    async learnStopLoss(strategy, marketRegime) {
        // Adaptive stop-loss based on market conditions and strategy performance
        const volatility = await this.getMarketVolatility(marketRegime);
        const recentDrawdowns = this.getRecentDrawdowns(strategy);
        
        if (recentDrawdowns.length === 0) return 1.0;

        const avgDrawdown = recentDrawdowns.reduce((a, b) => a + b) / recentDrawdowns.length;
        const maxDrawdown = Math.max(...recentDrawdowns);
        
        // Widen stop-loss in high volatility, tighten after large drawdowns
        const stopLossAdjustment = (volatility * 0.2) - (maxDrawdown * 0.5);
        
        return Math.min(2.0, Math.max(0.5, 1.0 + stopLossAdjustment));
    }

    async learnExecutionTiming(strategy, marketRegime) {
        // Learn optimal execution timing
        const timingPerformance = this.getTimingPerformance(strategy);
        
        if (timingPerformance.length < 20) return 1.0;

        // Find timing patterns that led to better execution
        const successfulTiming = timingPerformance.filter(p => p.slippage < 0.001);
        if (successfulTiming.length === 0) return 1.0;

        const avgTiming = successfulTiming.reduce((sum, p) => sum + p.delay, 0) / successfulTiming.length;
        const baseTiming = strategy.parameters.executionDelay || 100;
        
        return avgTiming / baseTiming;
    }

    applyForgettingMechanism(adjustments, lastUpdate) {
        // Gradually forget old learnings that aren't reinforced
        const daysSinceUpdate = (Date.now() - lastUpdate) / (1000 * 60 * 60 * 24);
        const forgettingFactor = Math.exp(-daysSinceUpdate / 30); // 30-day half-life
        
        return Object.fromEntries(
            Object.entries(adjustments).map(([key, value]) => [
                key,
                1.0 + ((value - 1.0) * forgettingFactor)
            ])
        );
    }

    async recordStrategyPerformance(strategy, parameters, outcome, marketData) {
        // Record performance for continual learning
        const performanceRecord = {
            strategy: strategy.name,
            parameters: parameters,
            outcome: outcome,
            marketData: marketData,
            timestamp: Date.now(),
            regime: this.classifyMarketRegime(marketData)
        };

        this.performanceHistory.push(performanceRecord);
        
        // Keep history manageable
        if (this.performanceHistory.length > 10000) {
            this.performanceHistory = this.performanceHistory.slice(-5000);
        }

        // Update learning models
        await this.updateLearningModels(performanceRecord);
        
        // Evolve parameters based on performance
        await this.evolveParameters(strategy, performanceRecord);
    }

    async updateLearningModels(performanceRecord) {
        // Update various learning components
        await this.updateRiskModel(performanceRecord);
        await this.updateTimingModel(performanceRecord);
        await this.updateSizingModel(performanceRecord);
        
        // Reinforcement learning update
        await this.reinforcementUpdate(performanceRecord);
    }

    async reinforcementUpdate(performanceRecord) {
        // Reinforcement learning: strengthen good decisions, weaken bad ones
        const reward = this.calculateReward(performanceRecord);
        
        if (reward > 0) {
            // Positive reinforcement
            await this.strengthenSuccessfulPatterns(performanceRecord);
        } else {
            // Negative reinforcement
            await this.weakenUnsuccessfulPatterns(performanceRecord);
        }
    }

    calculateReward(performance) {
        // Comprehensive reward calculation
        const profitReward = performance.outcome.profit / performance.outcome.capital;
        const riskPenalty = -performance.outcome.maxDrawdown * 2;
        const consistencyBonus = performance.outcome.consistency * 0.1;
        
        return profitReward + riskPenalty + consistencyBonus;
    }

    async strengthenSuccessfulPatterns(performance) {
        // Strengthen neural pathways that led to success
        const learningRate = this.adaptationRate * (1 + performance.outcome.profit);
        await this.applyPositiveGradient(performance, learningRate);
    }

    async weakenUnsuccessfulPatterns(performance) {
        // Weaken neural pathways that led to failure
        const learningRate = this.adaptationRate * Math.abs(performance.outcome.profit);
        await this.applyNegativeGradient(performance, learningRate);
    }

    getRegimePerformance(strategy, regime) {
        return this.performanceHistory.filter(p => 
            p.strategy === strategy.name && 
            p.regime === regime &&
            p.timestamp > Date.now() - (30 * 24 * 60 * 60 * 1000) // Last 30 days
        );
    }

    getRecentPerformance(strategy, count) {
        return this.performanceHistory
            .filter(p => p.strategy === strategy.name)
            .slice(-count)
            .map(p => p.outcome);
    }

    getRecentDrawdowns(strategy) {
        const recent = this.getRecentPerformance(strategy, 100);
        return recent.map(p => p.maxDrawdown).filter(d => d > 0);
    }

    getTimingPerformance(strategy) {
        return this.performanceHistory
            .filter(p => p.strategy === strategy.name && p.outcome.executionMetrics)
            .map(p => p.outcome.executionMetrics);
    }

    async getMarketVolatility(regime) {
        // Get current market volatility estimate
        const regimeData = this.performanceHistory.filter(p => p.regime === regime);
        if (regimeData.length === 0) return 0.1; // Default
        
        const volatilities = regimeData.map(p => p.marketData.volatility || 0.1);
        return volatilities.reduce((a, b) => a + b) / volatilities.length;
    }

    classifyMarketRegime(marketData) {
        // Classify market regime based on data
        if (marketData.volatility > 0.25) return 'high_volatility';
        if (marketData.trendStrength > 0.7) return 'trending';
        if (marketData.trendStrength < 0.3) return 'ranging';
        return 'neutral';
    }

    validateParameters(parameters, strategy) {
        // Ensure parameters stay within safe bounds
        const validated = { ...parameters };
        
        // Risk limits
        validated.riskMultiplier = Math.min(2.0, Math.max(0.1, parameters.riskMultiplier));
        validated.maxPositionSize = Math.min(0.5, Math.max(0.01, parameters.maxPositionSize));
        validated.stopLoss = Math.min(0.2, Math.max(0.01, parameters.stopLoss));
        
        return validated;
    }

    async calculateParameterConfidence(strategy, parameters) {
        // Calculate confidence in learned parameters
        const similarConfigs = this.findSimilarConfigurations(strategy, parameters);
        if (similarConfigs.length === 0) return 0.5;

        const performanceScores = similarConfigs.map(c => c.outcome.profit / c.outcome.capital);
        const avgPerformance = performanceScores.reduce((a, b) => a + b) / performanceScores.length;
        
        return Math.min(1.0, Math.max(0.1, 0.5 + (avgPerformance * 2)));
    }

    findSimilarConfigurations(strategy, parameters) {
        return this.performanceHistory.filter(p => 
            p.strategy === strategy.name &&
            this.parameterSimilarity(p.parameters, parameters) > 0.8
        );
    }

    parameterSimilarity(params1, params2) {
        const keys = Object.keys(params1);
        const similarities = keys.map(key => {
            const val1 = params1[key];
            const val2 = params2[key];
            if (typeof val1 === 'number' && typeof val2 === 'number') {
                return 1 - Math.abs(val1 - val2) / Math.max(val1, val2);
            }
            return val1 === val2 ? 1 : 0;
        });
        
        return similarities.reduce((a, b) => a + b) / similarities.length;
    }

    getLearningSource(adjustments) {
        const sources = [];
        if (adjustments.riskMultiplier !== 1.0) sources.push('risk_learning');
        if (adjustments.positionSize !== 1.0) sources.push('sizing_learning');
        if (adjustments.stopLoss !== 1.0) sources.push('stoploss_learning');
        if (adjustments.timing !== 1.0) sources.push('timing_learning');
        
        return sources.length > 0 ? sources : ['baseline'];
    }

    calculateAdaptationLevel(adjustments) {
        const changes = Object.values(adjustments).map(val => Math.abs(val - 1.0));
        return changes.reduce((a, b) => a + b) / changes.length;
    }

    getLearningInsights() {
        const insights = {
            totalLearningExamples: this.performanceHistory.length,
            adaptationRate: this.adaptationRate,
            recentPerformance: this.getRecentPerformance('all', 100),
            parameterEvolution: this.parameterEvolution.slice(-100)
        };
        
        return insights;
    }
}

module.exports = StrategyOptimizer;
