// dydx-integration.js - dYdX Flash Loan Integration
// Reverse-engineered from dYdX Solo Margin & Perpetuals

const { ethers } = require('ethers');
const DYDX_ABI = require('./abis/dydx-solo'); // dYdX Solo Margin ABI

class dYdXIntegration {
    constructor(config) {
        this.provider = new ethers.providers.JsonRpcProvider(config.rpcUrl);
        this.signer = new ethers.Wallet(config.privateKey, this.provider);
        this.soloMarginAddress = config.soloMarginAddress || '0x1E0447b19BB6EcFdAe1e4AE1694b0C3659614e4e'; // dYdX Mainnet
        this.soloMargin = new ethers.Contract(this.soloMarginAddress, DYDX_ABI, this.signer);
        this.maxLoanSize = config.maxLoanSize || ethers.utils.parseEther('5000000'); // $5M default
    }

    async executeFlashLoan(flashLoanParams) {
        try {
            // dYdX validation
            const validation = await this.validateFlashLoan(flashLoanParams);
            if (!validation.valid) {
                throw new Error(`dYdX validation failed: ${validation.reason}`);
            }

            // Prepare dYdX operation
            const operation = await this.prepareDydxOperation(flashLoanParams);

            // Execute flash loan
            const tx = await this.soloMargin.operate([operation], []);

            const receipt = await tx.wait();

            return {
                success: true,
                transactionHash: receipt.transactionHash,
                gasUsed: receipt.gasUsed,
                premium: ethers.BigNumber.from(0), // dYdX has 0 premium
                provider: 'dydx',
                timestamp: Date.now()
            };

        } catch (error) {
            await this.handleDydxError(error, flashLoanParams);
            throw error;
        }
    }

    async validateFlashLoan(params) {
        // dYdX specific validation
        const checks = [
            this.validateAsset(params.asset),
            this.validateAmount(params.amount),
            await this.validateAccountHealth(params)
        ];

        const results = await Promise.all(checks);
        const failedCheck = results.find(check => !check.valid);

        return failedCheck || { valid: true };
    }

    async validateAsset(assetAddress) {
        // dYdX: Check if market is supported
        const marketId = await this.getMarketId(assetAddress);
        
        return {
            valid: marketId !== null,
            reason: marketId === null ? `Asset not supported by dYdX` : null
        };
    }

    async validateAmount(amount) {
        return {
            valid: amount.lte(this.maxLoanSize),
            reason: amount.gt(this.maxLoanSize) ? `Amount exceeds dYdX limit` : null
        };
    }

    async validateAccountHealth(params) {
        // dYdX: Account health validation
        const accountInfo = await this.soloMargin.getAccountBalance(
            this.signer.address,
            await this.getMarketId(params.asset)
        );

        return {
            valid: accountInfo.balance.gte(0),
            reason: accountInfo.balance.lt(0) ? 'Account would be undercollateralized' : null
        };
    }

    async prepareDydxOperation(flashLoanParams) {
        // dYdX operation structure
        return {
            primaryAccountOwner: this.signer.address,
            primaryAccountId: 0, // Default account
            operationType: 0, // Withdraw
            marketId: await this.getMarketId(flashLoanParams.asset),
            amount: {
                value: flashLoanParams.amount,
                sign: false // Positive for deposit, false for withdraw
            },
            otherAddress: flashLoanParams.receiverAddress,
            data: '0x' // Additional data
        };
    }

    async getMarketId(assetAddress) {
        // Map asset address to dYdX market ID
        const markets = {
            '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2': 0, // WETH
            '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48': 2, // USDC
            '0xdAC17F958D2ee523a2206206994597C13D831ec7': 3, // USDT
            '0x6B175474E89094C44Da98b954EedeAC495271d0F': 1  // DAI
        };

        return markets[assetAddress] !== undefined ? markets[assetAddress] : null;
    }

    async getLiquidityMetrics(asset) {
        // dYdX liquidity metrics
        const marketId = await this.getMarketId(asset);
        
        return {
            availableLiquidity: await this.getAvailableLiquidity(marketId),
            totalLiquidity: await this.getTotalLiquidity(marketId),
            utilizationRate: await this.getUtilizationRate(marketId),
            interestRate: ethers.BigNumber.from(0), // dYdX flash loans are free
            successRate: await this.getHistoricalSuccessRate(asset),
            capacity: await this.getAvailableCapacity(marketId)
        };
    }

    async getAvailableLiquidity(marketId) {
        // Get available liquidity for market
        // Simplified - would use dYdX API
        return ethers.utils.parseEther('10000000'); // Mock data
    }

    async getTotalLiquidity(marketId) {
        return ethers.utils.parseEther('50000000'); // Mock data
    }

    async getUtilizationRate(marketId) {
        return 0.65; // Mock 65% utilization
    }

    async getHistoricalSuccessRate(asset) {
        return 0.99; // dYdX has high success rate
    }

    async getAvailableCapacity(marketId) {
        const liquidity = await this.getAvailableLiquidity(marketId);
        return liquidity.div(2); // Conservative 50% usage
    }

    async handleDydxError(error, params) {
        console.error('dYdX flash loan execution failed:', {
            asset: params.asset,
            amount: ethers.utils.formatEther(params.amount),
            error: error.message
        });

        // dYdX specific error handling
        if (error.message.includes('OperationImpl: Bad operation')) {
            console.error('Invalid operation structure for dYdX');
        } else if (error.message.includes('AccountBalance: Account is undercollateralized')) {
            console.error('Account would be undercollateralized');
        }
    }
}

module.exports = dYdXIntegration;
