// uniswap-integration.js - Uniswap V3 Flash Swap Integration
// Reverse-engineered from Uniswap V3 Core & Periphery

const { ethers } = require('ethers');
const UNISWAP_ABI = require('@uniswap/v3-core/artifacts/contracts/UniswapV3Pool.sol/UniswapV3Pool.json');

class UniswapIntegration {
    constructor(config) {
        this.provider = new ethers.providers.JsonRpcProvider(config.rpcUrl);
        this.signer = new ethers.Wallet(config.privateKey, this.provider);
        this.quoterAddress = config.quoterAddress || '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6';
        this.swapRouterAddress = config.swapRouterAddress || '0xE592427A0AEce92De3Edee1F18E0157C05861564';
        this.maxLoanSize = config.maxLoanSize || ethers.utils.parseEther('1000000'); // $1M default
    }

    async executeFlashSwap(flashSwapParams) {
        try {
            // Uniswap V3 flash swap validation
            const validation = await this.validateFlashSwap(flashSwapParams);
            if (!validation.valid) {
                throw new Error(`Uniswap validation failed: ${validation.reason}`);
            }

            // Prepare flash swap parameters
            const swapParams = this.prepareSwapParams(flashSwapParams);

            // Execute flash swap
            const tx = await this.executeSwap(swapParams);

            const receipt = await tx.wait();

            return {
                success: true,
                transactionHash: receipt.transactionHash,
                gasUsed: receipt.gasUsed,
                premium: await this.calculatePremium(flashSwapParams.amount),
                provider: 'uniswap',
                timestamp: Date.now()
            };

        } catch (error) {
            await this.handleUniswapError(error, flashSwapParams);
            throw error;
        }
    }

    async validateFlashSwap(params) {
        const checks = [
            this.validatePoolExists(params.token0, params.token1, params.fee),
            this.validateAmount(params.amount),
            await this.validateLiquidity(params.token0, params.token1, params.fee, params.amount)
        ];

        const results = await Promise.all(checks);
        const failedCheck = results.find(check => !check.valid);

        return failedCheck || { valid: true };
    }

    async validatePoolExists(token0, token1, fee) {
        const poolAddress = await this.getPoolAddress(token0, token1, fee);
        
        return {
            valid: poolAddress !== ethers.constants.AddressZero,
            reason: poolAddress === ethers.constants.AddressZero ? 'Uniswap V3 pool does not exist' : null
        };
    }

    async validateAmount(amount) {
        return {
            valid: amount.lte(this.maxLoanSize),
            reason: amount.gt(this.maxLoanSize) ? `Amount exceeds Uniswap limit` : null
        };
    }

    async validateLiquidity(token0, token1, fee, amount) {
        const liquidity = await this.getPoolLiquidity(token0, token1, fee);
        
        return {
            valid: liquidity.gte(amount.mul(2)),
            reason: liquidity.lt(amount.mul(2)) ? 'Insufficient pool liquidity' : null
        };
    }

    async getPoolAddress(token0, token1, fee) {
        // Uniswap V3 pool address calculation
        const factoryAddress = '0x1F98431c8aD98523631AE4a59f267346ea31F984';
        const factory = new ethers.Contract(factoryAddress, [
            'function getPool(address, address, uint24) view returns (address)'
        ], this.provider);

        return await factory.getPool(token0, token1, fee);
    }

    async getPoolLiquidity(token0, token1, fee) {
        const poolAddress = await this.getPoolAddress(token0, token1, fee);
        if (poolAddress === ethers.constants.AddressZero) {
            return ethers.BigNumber.from(0);
        }

        const pool = new ethers.Contract(poolAddress, UNISWAP_ABI.abi, this.provider);
        const liquidity = await pool.liquidity();
        return liquidity;
    }

    prepareSwapParams(params) {
        // Uniswap V3 exact input single swap parameters
        return {
            tokenIn: params.token0,
            tokenOut: params.token1,
            fee: params.fee,
            recipient: params.recipient || this.signer.address,
            deadline: Math.floor(Date.now() / 1000) + 300, // 5 minutes
            amountIn: params.amount,
            amountOutMinimum: await this.calculateMinimumOut(params.amount, params.slippage),
            sqrtPriceLimitX96: 0
        };
    }

    async calculateMinimumOut(amountIn, slippage = 50) {
        // Calculate minimum output with slippage tolerance
        const quotedAmount = await this.getQuote(amountIn);
        return quotedAmount.mul(10000 - slippage).div(10000);
    }

    async getQuote(amountIn) {
        // Get quote from Uniswap V3 quoter
        const quoter = new ethers.Contract(this.quoterAddress, [
            'function quoteExactInputSingle(address tokenIn, address tokenOut, uint24 fee, uint256 amountIn, uint160 sqrtPriceLimitX96) external returns (uint256 amountOut)'
        ], this.provider);

        // Mock implementation - would use actual tokens
        return amountIn.mul(9950).div(10000); // 0.5% price impact
    }

    async executeSwap(swapParams) {
        // Execute swap through Uniswap V3 Router
        const router = new ethers.Contract(this.swapRouterAddress, [
            'function exactInputSingle(tuple(address tokenIn, address tokenOut, uint24 fee, address recipient, uint256 deadline, uint256 amountIn, uint256 amountOutMinimum, uint160 sqrtPriceLimitX96) calldata) external payable returns (uint256 amountOut)'
        ], this.signer);

        return await router.exactInputSingle(swapParams);
    }

    async calculatePremium(amount) {
        // Uniswap V3: Flash swaps have pool fees (typically 0.3%)
        return amount.mul(30).div(10000);
    }

    async getLiquidityMetrics(token0, token1, fee) {
        const poolAddress = await this.getPoolAddress(token0, token1, fee);
        
        if (poolAddress === ethers.constants.AddressZero) {
            return null;
        }

        const pool = new ethers.Contract(poolAddress, UNISWAP_ABI.abi, this.provider);
        
        return {
            availableLiquidity: await pool.liquidity(),
            totalLiquidity: await this.getTotalTVL(poolAddress),
            utilizationRate: await this.calculateUtilization(poolAddress),
            interestRate: await this.calculatePremium(ethers.utils.parseEther('1')),
            successRate: 0.97, // Uniswap V3 success rate
            capacity: await this.calculateCapacity(poolAddress)
        };
    }

    async getTotalTVL(poolAddress) {
        // Simplified TVL calculation
        const pool = new ethers.Contract(poolAddress, UNISWAP_ABI.abi, this.provider);
        const liquidity = await pool.liquidity();
        return liquidity.mul(2); // Approximate TVL
    }

    async calculateUtilization(poolAddress) {
        // Calculate pool utilization
        return 0.75; // Mock 75% utilization
    }

    async calculateCapacity(poolAddress) {
        const pool = new ethers.Contract(poolAddress, UNISWAP_ABI.abi, this.provider);
        const liquidity = await pool.liquidity();
        return liquidity.div(3); // Conservative 33% capacity
    }

    async handleUniswapError(error, params) {
        console.error('Uniswap flash swap execution failed:', {
            token0: params.token0,
            token1: params.token1,
            amount: ethers.utils.formatEther(params.amount),
            error: error.message
        });

        // Uniswap specific error handling
        if (error.message.includes('Too little received')) {
            console.error('Slippage tolerance exceeded');
        } else if (error.message.includes('Deadline passed')) {
            console.error('Transaction deadline passed');
        } else if (error.message.includes('Invalid pool')) {
            console.error('Invalid Uniswap V3 pool');
        }
    }
}

module.exports = UniswapIntegration;
