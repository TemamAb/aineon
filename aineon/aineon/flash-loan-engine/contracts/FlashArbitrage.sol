// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

// FlashArbitrage.sol - Main Flash Loan Arbitrage Contract
// Reverse-engineered from Aave V3, OpenZeppelin patterns

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

interface ILendingPool {
    function flashLoan(
        address receiverAddress,
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata modes,
        address onBehalfOf,
        bytes calldata params,
        uint16 referralCode
    ) external;
}

interface IUniswapV3Router {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }
    
    function exactInputSingle(ExactInputSingleParams calldata params) 
        external 
        payable 
        returns (uint256 amountOut);
}

contract FlashArbitrage is ReentrancyGuard, Ownable {
    using SafeERC20 for IERC20;
    
    // Constants
    address public constant AAVE_LENDING_POOL = 0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2;
    address public constant UNISWAP_ROUTER = 0xE592427A0AEce92De3Edee1F18E0157C05861564;
    
    // State variables
    uint256 public constant FLASH_LOAN_PREMIUM = 9; // 0.09% for Aave
    uint256 public minProfitThreshold;
    address public feeReceiver;
    uint256 public performanceFee; // 20% performance fee
    
    // Events
    event FlashLoanExecuted(
        address indexed initiator,
        address[] assets,
        uint256[] amounts,
        uint256 premium,
        uint256 netProfit,
        uint256 timestamp
    );
    
    event ArbitrageFailed(
        address indexed initiator,
        address[] assets,
        uint256[] amounts,
        string reason,
        uint256 timestamp
    );
    
    event EmergencyWithdraw(
        address indexed token,
        uint256 amount,
        address recipient,
        uint256 timestamp
    );

    constructor(address _feeReceiver, uint256 _minProfitThreshold) {
        feeReceiver = _feeReceiver;
        minProfitThreshold = _minProfitThreshold;
        performanceFee = 2000; // 20% in basis points
    }

    /**
     * @dev Main flash loan execution function
     * @param assets Array of assets to flash loan
     * @param amounts Array of amounts to flash loan
     * @param modes Array of interest rate modes (0 for no debt)
     * @param arbitrageData Encoded data for arbitrage execution
     */
    function executeFlashLoanArbitrage(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata modes,
        bytes calldata arbitrageData
    ) external nonReentrant {
        require(assets.length == amounts.length, "Arrays length mismatch");
        require(assets.length == modes.length, "Arrays length mismatch");
        
        // Execute Aave flash loan
        ILendingPool(AAVE_LENDING_POOL).flashLoan(
            address(this),
            assets,
            amounts,
            modes,
            address(0),
            arbitrageData,
            0
        );
    }

    /**
     * @dev Aave flash loan callback function
     */
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external returns (bool) {
        require(msg.sender == AAVE_LENDING_POOL, "Caller must be lending pool");
        require(initiator == address(this), "Initiator must be this contract");
        
        // Decode arbitrage parameters
        (ArbitrageParams memory arbitrageParams) = abi.decode(params, (ArbitrageParams));
        
        // Execute arbitrage strategy
        uint256 netProfit = _executeArbitrageStrategy(assets, amounts, premiums, arbitrageParams);
        
        // Validate profitability
        require(netProfit >= minProfitThreshold, "Arbitrage not profitable");
        
        // Calculate and take performance fee
        uint256 feeAmount = (netProfit * performanceFee) / 10000;
        if (feeAmount > 0) {
            IERC20(arbitrageParams.profitToken).safeTransfer(feeReceiver, feeAmount);
            netProfit -= feeAmount;
        }
        
        // Transfer profit to initiator
        if (netProfit > 0) {
            IERC20(arbitrageParams.profitToken).safeTransfer(tx.origin, netProfit);
        }
        
        // Approve Aave to pull the loan amount + premium
        for (uint256 i = 0; i < assets.length; i++) {
            uint256 totalOwed = amounts[i] + premiums[i];
            IERC20(assets[i]).safeIncreaseAllowance(AAVE_LENDING_POOL, totalOwed);
        }
        
        emit FlashLoanExecuted(
            tx.origin,
            assets,
            amounts,
            _calculateTotalPremium(premiums),
            netProfit,
            block.timestamp
        );
        
        return true;
    }

    /**
     * @dev Execute arbitrage strategy (triangular, CEX/DEX, etc.)
     */
    function _executeArbitrageStrategy(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        ArbitrageParams memory params
    ) internal returns (uint256 netProfit) {
        // Store initial balance of profit token
        uint256 initialBalance = IERC20(params.profitToken).balanceOf(address(this));
        
        // Execute the arbitrage path
        for (uint256 i = 0; i < params.path.length - 1; i++) {
            _executeSwap(
                params.path[i],
                params.path[i + 1],
                params.amounts[i],
                params.minOutputs[i],
                params.fees[i]
            );
        }
        
        // Calculate profit
        uint256 finalBalance = IERC20(params.profitToken).balanceOf(address(this));
        uint256 grossProfit = finalBalance - initialBalance;
        
        // Calculate total premium cost
        uint256 totalPremiumCost = _calculatePremiumCost(assets, amounts, premiums, params.profitToken);
        
        netProfit = grossProfit - totalPremiumCost;
        
        require(netProfit >= 0, "Arbitrage resulted in loss");
    }

    /**
     * @dev Execute a single swap on Uniswap V3
     */
    function _executeSwap(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOutMinimum,
        uint24 fee
    ) internal {
        IERC20(tokenIn).safeIncreaseAllowance(UNISWAP_ROUTER, amountIn);
        
        IUniswapV3Router.ExactInputSingleParams memory swapParams = IUniswapV3Router.ExactInputSingleParams({
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            fee: fee,
            recipient: address(this),
            deadline: block.timestamp + 300, // 5 minute deadline
            amountIn: amountIn,
            amountOutMinimum: amountOutMinimum,
            sqrtPriceLimitX96: 0
        });
        
        IUniswapV3Router(UNISWAP_ROUTER).exactInputSingle(swapParams);
    }

    /**
     * @dev Calculate total premium cost in profit token terms
     */
    function _calculatePremiumCost(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address profitToken
    ) internal pure returns (uint256 totalCost) {
        // Simplified calculation - would use oracle for accurate pricing
        for (uint256 i = 0; i < assets.length; i++) {
            // Assume 1:1 conversion for stablecoins, would use oracle in production
            if (assets[i] == profitToken) {
                totalCost += premiums[i];
            } else {
                // Estimate conversion - in production would use price oracle
                totalCost += premiums[i];
            }
        }
        return totalCost;
    }

    /**
     * @dev Calculate total premium paid
     */
    function _calculateTotalPremium(uint256[] calldata premiums) internal pure returns (uint256 total) {
        for (uint256 i = 0; i < premiums.length; i++) {
            total += premiums[i];
        }
        return total;
    }

    /**
     * @dev Emergency withdraw function for owner
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        IERC20(token).safeTransfer(owner(), amount);
        emit EmergencyWithdraw(token, amount, owner(), block.timestamp);
    }

    /**
     * @dev Update minimum profit threshold
     */
    function setMinProfitThreshold(uint256 _threshold) external onlyOwner {
        minProfitThreshold = _threshold;
    }

    /**
     * @dev Update performance fee
     */
    function setPerformanceFee(uint256 _fee) external onlyOwner {
        require(_fee <= 5000, "Fee too high"); // Max 50%
        performanceFee = _fee;
    }

    /**
     * @dev Update fee receiver
     */
    function setFeeReceiver(address _receiver) external onlyOwner {
        require(_receiver != address(0), "Invalid address");
        feeReceiver = _receiver;
    }

    // Structs
    struct ArbitrageParams {
        address profitToken;
        address[] path;
        uint256[] amounts;
        uint256[] minOutputs;
        uint24[] fees;
    }

    // Receive ETH if needed
    receive() external payable {}
}
