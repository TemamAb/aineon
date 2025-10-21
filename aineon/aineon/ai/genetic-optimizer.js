// genetic-optimizer.js - Genetic Algorithm for Strategy Optimization
// Reverse-engineered from DEAP, JGAP, Watchmaker patterns

const { GeneticAlgorithm: BaseGA } = require('genetic-algorithm');

class GeneticOptimizer {
    constructor(config) {
        this.config = {
            populationSize: config.populationSize || 100,
            generations: config.generations || 50,
            mutationRate: config.mutationRate || 0.01,
            crossoverRate: config.crossoverRate || 0.8,
            elitismCount: config.elitismCount || 5,
            ...config
        };
        
        this.algorithm = new BaseGA(this.config);
        this.optimizationHistory = [];
        this.convergenceTracker = new ConvergenceTracker();
    }

    async initialize() {
        console.log('í·¬ Initializing Genetic Optimizer...');
        await this.initializeGeneticOperators();
        console.log('âœ… Genetic Optimizer initialized');
    }

    async optimizeStrategy(strategy, fitnessFunction, constraints) {
        // DEAP-inspired genetic optimization
        console.log(`ï¿½ï¿½ Starting genetic optimization for ${strategy.name}`);
        
        const parameterSpace = this.defineParameterSpace(strategy);
        const population = await this.initializePopulation(parameterSpace);
        
        let bestSolution = null;
        let bestFitness = -Infinity;
        
        for (let generation = 0; generation < this.config.generations; generation++) {
            const generationResult = await this.runGeneration(
                population, 
                fitnessFunction, 
                constraints, 
                generation
            );
            
            if (generationResult.bestFitness > bestFitness) {
                bestFitness = generationResult.bestFitness;
                bestSolution = generationResult.bestIndividual;
            }
            
            // Check for convergence
            if (this.convergenceTracker.hasConverged(generationResult)) {
                console.log(`âœ“ Convergence reached at generation ${generation}`);
                break;
            }
            
            // Evolve to next generation
            population.individuals = await this.createNextGeneration(
                population, 
                generationResult.fitnessScores
            );
            
            this.recordGenerationStats(generation, generationResult);
        }
        
        return this.finalizeOptimization(bestSolution, strategy);
    }

    async runGeneration(population, fitnessFunction, constraints, generation) {
        // Evaluate fitness for all individuals in population
        const fitnessScores = await Promise.all(
            population.individuals.map(individual =>
                this.evaluateIndividualFitness(individual, fitnessFunction, constraints)
            )
        );
        
        // Select best individuals
        const bestIndividual = this.selectBestIndividual(population.individuals, fitnessScores);
        const bestFitness = Math.max(...fitnessScores);
        
        return {
            generation,
            bestIndividual,
            bestFitness,
            averageFitness: fitnessScores.reduce((a, b) => a + b, 0) / fitnessScores.length,
            fitnessScores,
            diversity: this.calculatePopulationDiversity(population.individuals)
        };
    }

    async evaluateIndividualFitness(individual, fitnessFunction, constraints) {
        try {
            // Validate individual against constraints
            if (!this.validateIndividual(individual, constraints)) {
                return -Infinity;
            }
            
            // Evaluate fitness
            const fitness = await fitnessFunction(individual.parameters);
            
            // Apply penalty for constraint violations
            const penalty = this.calculateConstraintPenalty(individual, constraints);
            
            return Math.max(0, fitness - penalty);
            
        } catch (error) {
            console.warn('Individual evaluation failed:', error.message);
            return -Infinity;
        }
    }

    async createNextGeneration(population, fitnessScores) {
        // Create new generation using genetic operators
        const newGeneration = [];
        
        // Elitism: Carry best individuals to next generation
        const elites = this.selectElites(population.individuals, fitnessScores);
        newGeneration.push(...elites);
        
        // Create offspring through selection, crossover, and mutation
        while (newGeneration.length < this.config.populationSize) {
            const parents = this.selectParents(population.individuals, fitnessScores);
            const offspring = await this.createOffspring(parents);
            newGeneration.push(offspring);
        }
        
        return newGeneration.slice(0, this.config.populationSize);
    }

    selectElites(individuals, fitnessScores, count = this.config.elitismCount) {
        // Select best individuals for elitism
        const indexed = individuals.map((ind, idx) => ({
            individual: ind,
            fitness: fitnessScores[idx]
        }));
        
        return indexed
            .sort((a, b) => b.fitness - a.fitness)
            .slice(0, count)
            .map(item => item.individual);
    }

    selectParents(individuals, fitnessScores) {
        // Tournament selection for parent selection
        const tournamentSize = Math.max(2, Math.floor(individuals.length * 0.1));
        
        const parent1 = this.tournamentSelection(individuals, fitnessScores, tournamentSize);
        const parent2 = this.tournamentSelection(individuals, fitnessScores, tournamentSize);
        
        return [parent1, parent2];
    }

    tournamentSelection(individuals, fitnessScores, tournamentSize) {
        // Tournament selection algorithm
        let bestIndividual = null;
        let bestFitness = -Infinity;
        
        for (let i = 0; i < tournamentSize; i++) {
            const randomIndex = Math.floor(Math.random() * individuals.length);
            const fitness = fitnessScores[randomIndex];
            
            if (fitness > bestFitness) {
                bestFitness = fitness;
                bestIndividual = individuals[randomIndex];
            }
        }
        
        return bestIndividual;
    }

    async createOffspring(parents) {
        // Create offspring through crossover and mutation
        const [parent1, parent2] = parents;
        
        // Crossover
        const childGenes = await this.crossover(parent1.parameters, parent2.parameters);
        
        // Mutation
        const mutatedGenes = await this.mutate(childGenes);
        
        return {
            parameters: mutatedGenes,
            generation: parent1.generation + 1,
            parentIds: [parent1.id, parent2.id]
        };
    }

    async crossover(parent1Params, parent2Params) {
        // Multi-point crossover
        if (Math.random() > this.config.crossoverRate) {
            return Math.random() > 0.5 ? parent1Params : parent2Params;
        }
        
        const childParams = {};
        const keys = Object.keys(parent1Params);
        
        for (const key of keys) {
            if (Math.random() > 0.5) {
                childParams[key] = parent1Params[key];
            } else {
                childParams[key] = parent2Params[key];
            }
        }
        
        return childParams;
    }

    async mutate(parameters) {
        // Gaussian mutation for continuous parameters
        const mutated = { ...parameters };
        
        for (const [key, value] of Object.entries(parameters)) {
            if (Math.random() < this.config.mutationRate) {
                if (typeof value === 'number') {
                    // Gaussian mutation with 5% standard deviation
                    const mutation = this.gaussianRandom(0, Math.abs(value) * 0.05);
                    mutated[key] = Math.max(0, value + mutation);
                } else if (typeof value === 'boolean') {
                    // Flip boolean values
                    mutated[key] = !value;
                }
            }
        }
        
        return mutated;
    }

    defineParameterSpace(strategy) {
        // Define parameter search space based on strategy type
        const spaces = {
            flashLoanArbitrage: {
                minProfitThreshold: { type: 'continuous', min: 0.001, max: 0.05 },
                maxSlippage: { type: 'continuous', min: 0.001, max: 0.02 },
                timeoutSeconds: { type: 'integer', min: 10, max: 120 },
                positionSize: { type: 'continuous', min: 0.1, max: 1.0 }
            },
            marketMaking: {
                spread: { type: 'continuous', min: 0.0005, max: 0.01 },
                inventoryRisk: { type: 'continuous', min: 0.1, max: 2.0 },
                rebalanceFrequency: { type: 'integer', min: 1, max: 60 }
            },
            statisticalArbitrage: {
                zScoreEntry: { type: 'continuous', min: 1.0, max: 3.0 },
                zScoreExit: { type: 'continuous', min: 0.1, max: 1.0 },
                lookbackPeriod: { type: 'integer', min: 10, max: 200 }
            }
        };
        
        return spaces[strategy.type] || spaces.flashLoanArbitrage;
    }

    async initializePopulation(parameterSpace) {
        // Initialize random population within parameter space
        const individuals = [];
        
        for (let i = 0; i < this.config.populationSize; i++) {
            const parameters = {};
            
            for (const [param, spec] of Object.entries(parameterSpace)) {
                if (spec.type === 'continuous') {
                    parameters[param] = spec.min + Math.random() * (spec.max - spec.min);
                } else if (spec.type === 'integer') {
                    parameters[param] = Math.floor(spec.min + Math.random() * (spec.max - spec.min + 1));
                } else if (spec.type === 'boolean') {
                    parameters[param] = Math.random() > 0.5;
                }
            }
            
            individuals.push({
                id: this.generateIndividualId(),
                parameters,
                generation: 0
            });
        }
        
        return { individuals, parameterSpace };
    }

    validateIndividual(individual, constraints) {
        // Validate individual against constraints
        if (!constraints) return true;
        
        for (const [param, value] of Object.entries(individual.parameters)) {
            const constraint = constraints[param];
            if (constraint) {
                if (value < constraint.min || value > constraint.max) {
                    return false;
                }
            }
        }
        
        return true;
    }

    calculateConstraintPenalty(individual, constraints) {
        // Calculate penalty for constraint violations
        let penalty = 0;
        
        if (!constraints) return penalty;
        
        for (const [param, value] of Object.entries(individual.parameters)) {
            const constraint = constraints[param];
            if (constraint) {
                if (value < constraint.min) {
                    penalty += (constraint.min - value) * 100;
                } else if (value > constraint.max) {
                    penalty += (value - constraint.max) * 100;
                }
            }
        }
        
        return penalty;
    }

    selectBestIndividual(individuals, fitnessScores) {
        // Select individual with highest fitness
        let bestIndex = 0;
        let bestFitness = fitnessScores[0];
        
        for (let i = 1; i < fitnessScores.length; i++) {
            if (fitnessScores[i] > bestFitness) {
                bestFitness = fitnessScores[i];
                bestIndex = i;
            }
        }
        
        return individuals[bestIndex];
    }

    calculatePopulationDiversity(individuals) {
        // Calculate population diversity
        if (individuals.length <= 1) return 1;
        
        let totalDistance = 0;
        let comparisons = 0;
        
        for (let i = 0; i < individuals.length; i++) {
            for (let j = i + 1; j < individuals.length; j++) {
                const distance = this.calculateIndividualDistance(
                    individuals[i], 
                    individuals[j]
                );
                totalDistance += distance;
                comparisons++;
            }
        }
        
        return totalDistance / comparisons;
    }

    calculateIndividualDistance(ind1, ind2) {
        // Calculate distance between two individuals
        let distance = 0;
        const keys = Object.keys(ind1.parameters);
        
        for (const key of keys) {
            const val1 = ind1.parameters[key];
            const val2 = ind2.parameters[key];
            
            if (typeof val1 === 'number' && typeof val2 === 'number') {
                distance += Math.abs(val1 - val2);
            } else if (val1 !== val2) {
                distance += 1;
            }
        }
        
        return distance / keys.length;
    }

    finalizeOptimization(bestSolution, strategy) {
        // Finalize optimization results
        const result = {
            strategy: strategy.name,
            optimizedParameters: bestSolution.parameters,
            fitness: bestSolution.fitness,
            generations: this.optimizationHistory.length,
            convergence: this.convergenceTracker.getConvergenceStats(),
            parameterSensitivity: this.analyzeParameterSensitivity(),
            optimizationHistory: this.optimizationHistory
        };
        
        // Record in optimization history
        this.optimizationHistory.push({
            strategy: strategy.name,
            timestamp: Date.now(),
            result
        });
        
        return result;
    }

    recordGenerationStats(generation, generationResult) {
        // Record generation statistics
        this.optimizationHistory.push({
            generation,
            bestFitness: generationResult.bestFitness,
            averageFitness: generationResult.averageFitness,
            diversity: generationResult.diversity,
            timestamp: Date.now()
        });
        
        // Update convergence tracker
        this.convergenceTracker.recordGeneration(generationResult);
    }

    async initializeGeneticOperators() {
        // Initialize genetic operators
        this.operators = {
            selection: new TournamentSelection(),
            crossover: new MultiPointCrossover(),
            mutation: new GaussianMutation()
        };
    }

    // Helper methods
    generateIndividualId() {
        return `ind_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    gaussianRandom(mean, stdDev) {
        // Generate Gaussian random number
        let u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        return mean + stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    analyzeParameterSensitivity() {
        // Analyze sensitivity of parameters to fitness
        // This would analyze how changes in parameters affect fitness
        return {
            // Mock sensitivity analysis
            minProfitThreshold: 0.8,
            maxSlippage: 0.6,
            timeoutSeconds: 0.3
        };
    }
}

// Supporting classes
class ConvergenceTracker {
    constructor() {
        this.generations = [];
        this.stagnationLimit = 10;
    }

    recordGeneration(generationResult) {
        this.generations.push(generationResult);
        
        // Keep only recent generations
        if (this.generations.length > 50) {
            this.generations = this.generations.slice(-50);
        }
    }

    hasConverged(currentGeneration) {
        if (this.generations.length < this.stagnationLimit) {
            return false;
        }
        
        // Check for fitness stagnation
        const recentGenerations = this.generations.slice(-this.stagnationLimit);
        const fitnessValues = recentGenerations.map(g => g.bestFitness);
        
        const improvement = Math.max(...fitnessValues) - Math.min(...fitnessValues);
        return improvement < 0.001; // Less than 0.1% improvement
    }

    getConvergenceStats() {
        if (this.generations.length === 0) {
            return { converged: false, generations: 0, improvement: 0 };
        }
        
        const first = this.generations[0];
        const last = this.generations[this.generations.length - 1];
        const improvement = last.bestFitness - first.bestFitness;
        
        return {
            converged: this.hasConverged(last),
            generations: this.generations.length,
            improvement,
            finalFitness: last.bestFitness
        };
    }
}

class TournamentSelection {
    select(population, fitnessScores, tournamentSize) {
        // Tournament selection implementation
        let bestIndex = Math.floor(Math.random() * population.length);
        let bestFitness = fitnessScores[bestIndex];
        
        for (let i = 1; i < tournamentSize; i++) {
            const candidateIndex = Math.floor(Math.random() * population.length);
            const candidateFitness = fitnessScores[candidateIndex];
            
            if (candidateFitness > bestFitness) {
                bestIndex = candidateIndex;
                bestFitness = candidateFitness;
            }
        }
        
        return population[bestIndex];
    }
}

class MultiPointCrossover {
    crossover(parent1, parent2, points = 2) {
        // Multi-point crossover implementation
        const child = { ...parent1 };
        const keys = Object.keys(parent1);
        const crossoverPoints = this.generateCrossoverPoints(keys.length, points);
        
        let useParent1 = true;
        let pointIndex = 0;
        
        for (let i = 0; i < keys.length; i++) {
            if (i >= crossoverPoints[pointIndex]) {
                useParent1 = !useParent1;
                pointIndex++;
            }
            
            if (!useParent1) {
                child[keys[i]] = parent2[keys[i]];
            }
        }
        
        return child;
    }

    generateCrossoverPoints(length, points) {
        const crossoverPoints = [];
        for (let i = 0; i < points; i++) {
            crossoverPoints.push(Math.floor(Math.random() * length));
        }
        return crossoverPoints.sort((a, b) => a - b);
    }
}

class GaussianMutation {
    mutate(parameters, mutationRate = 0.01, stdDev = 0.05) {
        // Gaussian mutation implementation
        const mutated = { ...parameters };
        
        for (const [key, value] of Object.entries(parameters)) {
            if (Math.random() < mutationRate && typeof value === 'number') {
                const mutation = this.gaussianRandom(0, Math.abs(value) * stdDev);
                mutated[key] = Math.max(0, value + mutation);
            }
        }
        
        return mutated;
    }

    gaussianRandom(mean, stdDev) {
        let u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        return mean + stdDev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
}

module.exports = GeneticOptimizer;
