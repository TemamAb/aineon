// capital-allocator.js - Dynamic Capital Allocation
// Reverse-engineered from Yearn Vaults, Balancer, Aave Pool strategies

const { ethers } = require('ethers');

class CapitalAllocator {
    constructor(config) {
        this.providers = config.providers;
        this.totalCapital = ethers.utils.parseEther('100000000'); // $100M
        this.allocationStrategy = config.strategy || 'risk-weighted';
        this.rebalancingInterval = config.rebalancingInterval || 3600000; // 1 hour
    }

    async calculateOptimalAllocation(opportunities) {
        // Yearn Vaults pattern: Risk-weighted capital allocation
        const allocations = await Promise.all(
            opportunities.map(opp => this.calculateOpportunityAllocation(opp))
        );

        const totalWeight = allocations.reduce((sum, alloc) => sum + alloc.weight, 0);
        
        return allocations.map(alloc => ({
            opportunity: alloc.opportunity,
            allocation: alloc.weight / totalWeight * this.totalCapital,
            provider: alloc.provider,
            riskScore: alloc.riskScore
        }));
    }

    async calculateOpportunityAllocation(opportunity) {
        // Balancer-inspired weighted scoring
        const riskScore = await this.calculateRiskScore(opportunity);
        const profitabilityScore = await this.calculateProfitabilityScore(opportunity);
        const liquidityScore = await this.calculateLiquidityScore(opportunity);

        const weight = this.calculateWeight(riskScore, profitabilityScore, liquidityScore);

        return {
            opportunity,
            weight,
            riskScore,
            provider: await this.selectProviderForOpportunity(opportunity)
        };
    }

    async calculateRiskScore(opportunity) {
        // Aave V3 risk assessment patterns
        const volatility = await this.getAssetVolatility(opportunity.asset);
        const liquidityRisk = await this.getLiquidityRisk(opportunity.asset);
        const executionRisk = await this.getExecutionRisk(opportunity);
        
        return Math.max(0, 1 - (volatility * 0.4 + liquidityRisk * 0.3 + executionRisk * 0.3));
    }

    async calculateProfitabilityScore(opportunity) {
        // Yearn Vaults profitability calculation
        const expectedProfit = opportunity.expectedProfit;
        const maxDrawdown = opportunity.maxDrawdown || 0.1; // 10% default
        
        return Math.min(1, expectedProfit / (maxDrawdown * opportunity.amount));
    }

    async calculateLiquidityScore(opportunity) {
        // Aave V3 liquidity depth assessment
        const providerLiquidity = await this.getProviderLiquidity(opportunity.providers);
        const requiredLiquidity = opportunity.amount;
        
        return Math.min(1, providerLiquidity / requiredLiquidity);
    }

    calculateWeight(riskScore, profitabilityScore, liquidityScore) {
        // Balancer-style weighted combination
        const weights = {
            risk: 0.4,
            profit: 0.4,
            liquidity: 0.2
        };

        return (
            riskScore * weights.risk +
            profitabilityScore * weights.profit +
            liquidityScore * weights.liquidity
        );
    }

    async selectProviderForOpportunity(opportunity) {
        // Aave V3 provider selection based on capital efficiency
        const providerMetrics = await Promise.all(
            opportunity.providers.map(async provider => ({
                provider,
                metrics: await this.getProviderMetrics(provider, opportunity.asset)
            }))
        );

        return providerMetrics.sort((a, b) => 
            b.metrics.capitalEfficiency - a.metrics.capitalEfficiency
        )[0].provider;
    }

    async getProviderMetrics(provider, asset) {
        // Aave V3 provider performance metrics
        return {
            capitalEfficiency: await this.calculateCapitalEfficiency(provider, asset),
            successRate: await this.getProviderSuccessRate(provider),
            liquidityDepth: await this.getProviderLiquidity(provider, asset)
        };
    }

    async calculateCapitalEfficiency(provider, asset) {
        // Yearn Vaults capital efficiency calculation
        const historicalReturns = await this.getHistoricalReturns(provider, asset);
        const riskAdjustedReturns = historicalReturns.map(return_ => return_ / this.getRiskFactor(asset));
        
        return riskAdjustedReturns.reduce((sum, ret) => sum + ret, 0) / historicalReturns.length;
    }

    async rebalanceAllocations(currentAllocations, newOpportunities) {
        // Balancer-style portfolio rebalancing
        const targetAllocations = await this.calculateOptimalAllocation(newOpportunities);
        
        return this.calculateRebalancingTrades(currentAllocations, targetAllocations);
    }

    calculateRebalancingTrades(current, target) {
        // Calculate required trades to reach target allocation
        return target.map(targetAlloc => {
            const currentAlloc = current.find(c => 
                c.opportunity.id === targetAlloc.opportunity.id
            ) || { allocation: ethers.BigNumber.from(0) };

            const difference = targetAlloc.allocation.sub(currentAlloc.allocation);
            
            return {
                opportunity: targetAlloc.opportunity,
                tradeAmount: difference,
                direction: difference.gt(0) ? 'increase' : 'decrease'
            };
        }).filter(trade => !trade.tradeAmount.isZero());
    }
}

module.exports = CapitalAllocator;
