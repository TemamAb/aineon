// strategicAI.js - Main AI Decision Engine WITH CONTINUAL LEARNING
// Reverse-engineered from TensorFlow.js + Continual Learning patterns

const tf = require('@tensorflow/tfjs');
const { NeuralNetwork } = require('./models/pattern-recognition');

class StrategicAI {
    constructor(config) {
        this.config = config;
        this.isInitialized = false;
        this.models = new Map();
        this.decisionHistory = [];
        this.performanceMetrics = new ContinualPerformanceTracker();
        this.experienceBuffer = new ExperienceBuffer(5000); // Store 5000 decisions
        this.adaptationEngine = new AdaptationEngine();
        this.learningCycles = 0;
    }

    async initialize() {
        try {
            console.log('í·  Initializing Strategic AI Engine with Continual Learning...');
            
            // Load all AI models
            await this.loadModels();
            
            // Load learned weights from previous sessions
            await this.loadLearnedWeights();
            
            // Warm up models with sample data
            await this.warmUpModels();
            
            // Start continuous learning loop
            this.startLearningLoop();
            
            this.isInitialized = true;
            console.log('âœ… Strategic AI Engine initialized with Continual Learning');
            
        } catch (error) {
            console.error('âŒ Strategic AI initialization failed:', error);
            throw error;
        }
    }

    async loadLearnedWeights() {
        // Load previously learned weights for continual learning
        try {
            const learnedWeights = await this.loadFromStorage('strategic_ai_weights');
            if (learnedWeights) {
                await this.applyLearnedWeights(learnedWeights);
                console.log('âœ… Loaded previously learned weights');
            }
        } catch (error) {
            console.log('âš ï¸ No previous weights found, starting fresh');
        }
    }

    async applyLearnedWeights(weights) {
        for (const [modelName, modelWeights] of Object.entries(weights)) {
            if (this.models.has(modelName)) {
                const model = this.models.get(modelName);
                await model.setWeights(modelWeights);
            }
        }
    }

    startLearningLoop() {
        // Continuous learning in background
        this.learningInterval = setInterval(async () => {
            await this.continuousLearningCycle();
        }, this.config.learningInterval || 300000); // 5 minutes
        
        console.log('í´„ Started continuous learning loop');
    }

    async continuousLearningCycle() {
        if (this.experienceBuffer.size() < 100) return; // Need minimum data
        
        try {
            this.learningCycles++;
            
            // Get batch for learning
            const learningBatch = this.experienceBuffer.getLearningBatch(64);
            
            // Online learning update
            await this.onlineModelUpdate(learningBatch);
            
            // Adaptive learning rate adjustment
            this.adaptationEngine.adjustLearningRates(this.performanceMetrics);
            
            // Save learned weights periodically
            if (this.learningCycles % 10 === 0) {
                await this.saveLearnedWeights();
            }
            
            console.log(`í´ Learning cycle ${this.learningCycles} completed`);
            
        } catch (error) {
            console.error('Continuous learning cycle failed:', error);
        }
    }

    async onlineModelUpdate(learningBatch) {
        for (const [modelName, model] of this.models) {
            const modelBatch = learningBatch.filter(exp => 
                exp.decision.metadata.modelPredictions[modelName]
            );
            
            if (modelBatch.length > 0) {
                await this.updateModelWeights(modelName, model, modelBatch);
            }
        }
    }

    async updateModelWeights(modelName, model, batch) {
        // Online weight updates using gradient descent
        const inputs = this.prepareBatchInputs(batch);
        const targets = this.prepareBatchTargets(batch, modelName);
        
        // Custom training step for continual learning
        const history = await model.fit(inputs, targets, {
            epochs: 1,
            batchSize: Math.min(32, batch.length),
            verbose: 0,
            shuffle: true
        });
        
        // Track learning progress
        this.performanceMetrics.recordModelUpdate(modelName, history.history.loss[0]);
        
        // Clean up tensors
        inputs.dispose();
        targets.dispose();
    }

    prepareBatchInputs(batch) {
        const allFeatures = batch.map(exp => 
            this.prepareInputData(exp.marketData, exp.portfolioState, exp.riskProfile).arraySync()[0]
        );
        return tf.tensor2d(allFeatures, [batch.length, allFeatures[0].length]);
    }

    prepareBatchTargets(batch, modelName) {
        // Prepare targets based on actual outcomes for supervised learning
        const targets = batch.map(exp => {
            const actualOutcome = exp.outcome;
            const predicted = exp.decision.metadata.modelPredictions[modelName];
            
            // Calculate target adjustment based on outcome
            return this.calculateTargetAdjustment(predicted, actualOutcome, modelName);
        });
        
        return tf.tensor2d(targets, [batch.length, targets[0].length]);
    }

    calculateTargetAdjustment(predicted, actualOutcome, modelName) {
        // Adjust targets based on actual performance (reward/punishment)
        switch (modelName) {
            case 'opportunity_scoring':
                const profitSignal = actualOutcome.profit > 0 ? 1 : -1;
                return predicted.map(p => Math.max(0, Math.min(1, p + (profitSignal * 0.1))));
                
            case 'risk_assessment':
                const riskSignal = actualOutcome.maxDrawdown > 0.1 ? 1 : 0;
                return this.adjustRiskTargets(predicted, riskSignal);
                
            default:
                return predicted; // No adjustment for other models
        }
    }

    adjustRiskTargets(predicted, riskSignal) {
        // Adjust risk assessment based on actual drawdowns
        const adjusted = [...predicted];
        if (riskSignal > 0) {
            // Increase high risk probability if actual drawdown was large
            adjusted[2] = Math.min(1, adjusted[2] + 0.2);
            adjusted[0] = Math.max(0, adjusted[0] - 0.1);
        }
        return adjusted;
    }

    async makeStrategicDecision(marketData, portfolioState, riskProfile) {
        if (!this.isInitialized) {
            throw new Error('Strategic AI not initialized');
        }

        try {
            // Prepare input data
            const inputTensor = this.prepareInputData(marketData, portfolioState, riskProfile);
            
            // Get predictions from all models
            const predictions = await this.getModelPredictions(inputTensor);
            
            // Aggregate decisions with confidence scoring
            const decision = this.aggregateDecisions(predictions, marketData);
            
            // Validate decision against risk parameters
            const validatedDecision = this.validateDecision(decision, riskProfile);
            
            // Add continual learning metadata
            validatedDecision.learningContext = {
                experienceId: this.experienceBuffer.nextId(),
                modelVersions: this.getModelVersions(),
                learningCycle: this.learningCycles
            };
            
            return validatedDecision;

        } catch (error) {
            console.error('Strategic decision failed:', error);
            return this.getFallbackDecision(marketData, riskProfile);
        }
    }

    async recordDecisionOutcome(decision, outcome, marketData, portfolioState, riskProfile) {
        // Record complete experience for continual learning
        const experience = {
            decision,
            outcome,
            marketData,
            portfolioState,
            riskProfile,
            timestamp: Date.now(),
            learningCycle: this.learningCycles
        };
        
        // Add to experience buffer
        this.experienceBuffer.add(experience);
        
        // Update performance metrics
        this.performanceMetrics.recordDecisionOutcome(decision, outcome);
        
        // Immediate learning from significant outcomes
        if (Math.abs(outcome.profit) > outcome.capital * 0.02) { // 2%+ moves
            await this.immediateLearning(experience);
        }
        
        console.log(`í³Š Recorded decision outcome: ${outcome.profit >= 0 ? 'âœ…' : 'âŒ'} ${outcome.profit}`);
    }

    async immediateLearning(experience) {
        // Learn immediately from significant outcomes
        const learningRate = this.calculateImmediateLearningRate(experience.outcome);
        
        for (const [modelName, model] of this.models) {
            await this.applyImmediateUpdate(modelName, model, experience, learningRate);
        }
    }

    calculateImmediateLearningRate(outcome) {
        // Higher learning rate for surprising outcomes
        const surprise = Math.abs(outcome.expectedProfit - outcome.actualProfit) / outcome.capital;
        return Math.min(0.1, 0.01 * (1 + surprise * 10));
    }

    async applyImmediateUpdate(modelName, model, experience, learningRate) {
        // Apply immediate weight update using custom optimizer
        const input = this.prepareInputData(
            experience.marketData, 
            experience.portfolioState, 
            experience.riskProfile
        );
        
        const actual = this.calculateTargetAdjustment(
            experience.decision.metadata.modelPredictions[modelName],
            experience.outcome,
            modelName
        );
        
        const target = tf.tensor2d([actual], [1, actual.length]);
        
        // Custom gradient update with higher learning rate
        await this.customGradientUpdate(model, input, target, learningRate);
        
        input.dispose();
        target.dispose();
    }

    async customGradientUpdate(model, input, target, learningRate) {
        // Custom gradient descent for immediate learning
        const optimizer = tf.train.sgd(learningRate);
        
        return optimizer.minimize(() => {
            const prediction = model.predict(input);
            const loss = tf.losses.meanSquaredError(target, prediction);
            return loss;
        }, true);
    }

    getModelVersions() {
        const versions = {};
        for (const [name] of this.models) {
            versions[name] = {
                learningCycle: this.learningCycles,
                updateCount: this.performanceMetrics.getModelUpdateCount(name),
                performance: this.performanceMetrics.getModelPerformance(name)
            };
        }
        return versions;
    }

    async saveLearnedWeights() {
        const weights = {};
        for (const [modelName, model] of this.models) {
            weights[modelName] = await model.getWeights();
        }
        
        await this.saveToStorage('strategic_ai_weights', weights);
        console.log('í²¾ Saved learned weights to storage');
    }

    // Storage methods (would integrate with database)
    async loadFromStorage(key) {
        // Mock implementation - would use actual storage
        return null;
    }

    async saveToStorage(key, data) {
        // Mock implementation - would use actual storage
        return true;
    }

    getLearningMetrics() {
        return {
            learningCycles: this.learningCycles,
            experienceBufferSize: this.experienceBuffer.size(),
            performance: this.performanceMetrics.getSummary(),
            modelVersions: this.getModelVersions(),
            adaptation: this.adaptationEngine.getStatus()
        };
    }

    // ... (keep all the original methods from previous version)
    prepareInputData(marketData, portfolioState, riskProfile) {
        // Implementation from previous version
        const features = [];
        features.push(...this.normalizeMarketData(marketData));
        features.push(...this.normalizePortfolioData(portfolioState));
        features.push(...this.normalizeRiskProfile(riskProfile));
        return tf.tensor2d([features], [1, features.length]);
    }

    normalizeMarketData(marketData) {
        // Implementation from previous version
        const normalized = [];
        normalized.push(this.minMaxNormalize(marketData.priceChange24h, -0.1, 0.1));
        normalized.push(this.minMaxNormalize(marketData.volumeChange24h, -0.5, 0.5));
        normalized.push(this.minMaxNormalize(marketData.volatility30d, 0, 0.5));
        normalized.push(this.minMaxNormalize(marketData.volatility7d, 0, 0.3));
        normalized.push(marketData.isBullish ? 1 : 0);
        normalized.push(marketData.isConsolidating ? 1 : 0);
        normalized.push(this.minMaxNormalize(marketData.liquidityScore, 0, 1));
        return normalized;
    }

    normalizePortfolioData(portfolio) {
        // Implementation from previous version
        const normalized = [];
        normalized.push(this.minMaxNormalize(portfolio.totalValue, 0, 100000000));
        normalized.push(this.minMaxNormalize(portfolio.availableCash, 0, 10000000));
        normalized.push(this.minMaxNormalize(portfolio.currentExposure, 0, 0.5));
        normalized.push(this.minMaxNormalize(portfolio.avgPositionSize, 0, 1000000));
        return normalized;
    }

    normalizeRiskProfile(riskProfile) {
        // Implementation from previous version
        const normalized = [];
        normalized.push(riskProfile.tolerance);
        normalized.push(riskProfile.maxDrawdown / 100);
        normalized.push(riskProfile.targetReturn / 100);
        return normalized;
    }

    minMaxNormalize(value, min, max) {
        return (value - min) / (max - min);
    }

    async getModelPredictions(inputTensor) {
        // Implementation from previous version
        const predictions = {};
        for (const [name, model] of this.models) {
            const prediction = await model.predict(inputTensor).data();
            predictions[name] = Array.from(prediction);
        }
        inputTensor.dispose();
        return predictions;
    }

    aggregateDecisions(predictions, marketData) {
        // Implementation from previous version
        const marketRegime = this.interpretMarketRegime(predictions.market_regime);
        const opportunityScore = this.calculateOpportunityScore(predictions.opportunity_scoring);
        const riskLevel = this.interpretRiskLevel(predictions.risk_assessment);
        const allocation = this.calculateAllocation(predictions.capital_allocation, marketData);
        
        return {
            action: this.determineAction(opportunityScore, riskLevel, marketRegime),
            confidence: this.calculateConfidence(predictions),
            allocation: allocation,
            riskLevel: riskLevel,
            marketRegime: marketRegime,
            opportunityScore: opportunityScore,
            timestamp: Date.now(),
            metadata: {
                modelPredictions: predictions,
                marketConditions: marketData
            }
        };
    }

    interpretMarketRegime(prediction) {
        // Implementation from previous version
        const [bullishProb, bearishProb, consolidatingProb] = prediction;
        if (bullishProb > 0.6) return 'bullish';
        if (bearishProb > 0.6) return 'bearish';
        if (consolidatingProb > 0.6) return 'consolidating';
        return 'transitional';
    }

    calculateOpportunityScore(prediction) {
        return Math.min(1, Math.max(0, prediction[0]));
    }

    interpretRiskLevel(prediction) {
        const [lowRisk, mediumRisk, highRisk] = prediction;
        if (highRisk > 0.7) return 'high';
        if (mediumRisk > 0.6) return 'medium';
        return 'low';
    }

    calculateAllocation(prediction, marketData) {
        const baseAllocation = prediction[0];
        const adjustedAllocation = this.adjustForMarketConditions(baseAllocation, marketData);
        return Math.min(0.5, Math.max(0.01, adjustedAllocation));
    }

    adjustForMarketConditions(allocation, marketData) {
        let adjustment = 1.0;
        if (marketData.volatility30d > 0.4) adjustment *= 0.7;
        if (marketData.liquidityScore < 0.3) adjustment *= 0.8;
        return allocation * adjustment;
    }

    determineAction(opportunityScore, riskLevel, marketRegime) {
        if (opportunityScore < 0.3 || riskLevel === 'high') return 'HOLD';
        if (opportunityScore > 0.7 && riskLevel === 'low') return 'AGGRESSIVE';
        if (opportunityScore > 0.5 && riskLevel === 'medium') return 'MODERATE';
        return 'CAUTIOUS';
    }

    calculateConfidence(predictions) {
        const confidences = Object.values(predictions).map(pred => {
            if (pred.length === 1) return Math.abs(pred[0] - 0.5) * 2;
            return Math.max(...pred);
        });
        return confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length;
    }

    validateDecision(decision, riskProfile) {
        const validated = { ...decision };
        if (riskProfile.maxDrawdown && decision.riskLevel === 'high') {
            validated.allocation = Math.min(validated.allocation, 0.1);
        }
        if (decision.confidence < 0.6) {
            validated.action = 'HOLD';
            validated.confidence = decision.confidence;
        }
        return validated;
    }

    getFallbackDecision(marketData, riskProfile) {
        return {
            action: 'HOLD',
            confidence: 0.3,
            allocation: 0.05,
            riskLevel: 'medium',
            marketRegime: 'unknown',
            opportunityScore: 0.3,
            timestamp: Date.now(),
            metadata: { isFallback: true, reason: 'AI system unavailable' }
        };
    }

    cleanup() {
        if (this.learningInterval) {
            clearInterval(this.learningInterval);
        }
        tf.disposeVariables();
        this.models.clear();
        this.isInitialized = false;
    }
}

// Continual Learning Support Classes
class ContinualPerformanceTracker {
    constructor() {
        this.metrics = {
            totalDecisions: 0,
            successfulDecisions: 0,
            averageConfidence: 0,
            modelUpdates: new Map(),
            decisionOutcomes: []
        };
    }

    recordDecisionOutcome(decision, outcome) {
        this.metrics.totalDecisions++;
        this.metrics.averageConfidence = 
            (this.metrics.averageConfidence * (this.metrics.totalDecisions - 1) + decision.confidence) 
            / this.metrics.totalDecisions;
        
        if (outcome.profit > 0) {
            this.metrics.successfulDecisions++;
        }
        
        this.metrics.decisionOutcomes.push({
            decision,
            outcome,
            timestamp: Date.now()
        });
        
        // Keep only recent outcomes
        if (this.metrics.decisionOutcomes.length > 1000) {
            this.metrics.decisionOutcomes = this.metrics.decisionOutcomes.slice(-500);
        }
    }

    recordModelUpdate(modelName, loss) {
        const updates = this.metrics.modelUpdates.get(modelName) || { count: 0, totalLoss: 0 };
        updates.count++;
        updates.totalLoss += loss;
        updates.averageLoss = updates.totalLoss / updates.count;
        this.metrics.modelUpdates.set(modelName, updates);
    }

    getModelUpdateCount(modelName) {
        const updates = this.metrics.modelUpdates.get(modelName);
        return updates ? updates.count : 0;
    }

    getModelPerformance(modelName) {
        const updates = this.metrics.modelUpdates.get(modelName);
        return updates ? { averageLoss: updates.averageLoss } : null;
    }

    getSummary() {
        const successRate = this.metrics.totalDecisions > 0 
            ? (this.metrics.successfulDecisions / this.metrics.totalDecisions) 
            : 0;
            
        return {
            totalDecisions: this.metrics.totalDecisions,
            successRate: successRate,
            averageConfidence: this.metrics.averageConfidence,
            modelUpdates: Object.fromEntries(this.metrics.modelUpdates)
        };
    }
}

class ExperienceBuffer {
    constructor(maxSize) {
        this.buffer = [];
        this.maxSize = maxSize;
        this.nextExperienceId = 1;
    }

    add(experience) {
        experience.id = this.nextExperienceId++;
        this.buffer.push(experience);
        
        if (this.buffer.length > this.maxSize) {
            this.buffer.shift();
        }
    }

    size() {
        return this.buffer.length;
    }

    getLearningBatch(size = 32) {
        if (this.buffer.length < size) {
            return this.buffer.slice();
        }
        
        // Prioritize recent and significant experiences
        const recent = this.buffer.slice(-Math.floor(size * 0.7));
        const significant = this.buffer
            .filter(exp => Math.abs(exp.outcome.profit) > exp.outcome.capital * 0.01)
            .slice(0, Math.floor(size * 0.3));
        
        return [...recent, ...significant].slice(0, size);
    }

    nextId() {
        return this.nextExperienceId;
    }
}

class AdaptationEngine {
    constructor() {
        this.learningRates = new Map();
        this.adaptationHistory = [];
    }

    adjustLearningRates(performanceMetrics) {
        const summary = performanceMetrics.getSummary();
        
        // Decrease learning rate if performance is good, increase if struggling
        const performanceFactor = summary.successRate > 0.6 ? 0.9 : 1.1;
        
        for (const [modelName] of this.learningRates) {
            const currentRate = this.learningRates.get(modelName) || 0.01;
            const newRate = Math.max(0.001, Math.min(0.1, currentRate * performanceFactor));
            this.learningRates.set(modelName, newRate);
        }
        
        this.adaptationHistory.push({
            timestamp: Date.now(),
            performanceFactor,
            successRate: summary.successRate
        });
    }

    getStatus() {
        return {
            learningRates: Object.fromEntries(this.learningRates),
            adaptationCount: this.adaptationHistory.length,
            recentPerformance: this.adaptationHistory.slice(-10)
        };
    }
}

module.exports = StrategicAI;
