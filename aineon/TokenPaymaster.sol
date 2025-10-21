// SPDX-License-Identifier: MIT
// PLATINUM SOURCES: ERC-4337, Gelato
// CONTINUAL LEARNING: Payment token learning, conversion optimization

pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openteppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

/**
 * @title TokenPaymaster
 * @dev ERC-4337 Paymaster that accepts token payments for gas
 * @notice Advanced paymaster with token conversions, price oracles, and gas optimization
 */
contract TokenPaymaster is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;
    using SafeMath for uint256;
    
    // ============ CONSTANTS ============
    uint256 public constant MAX_BPS = 10_000;
    uint256 public constant MIN_STAKE = 0.1 ether;
    
    // ============ STORAGE ============
    struct TokenConfig {
        bool isSupported;
        uint256 conversionRate; // tokens per ETH (scaled by 1e18)
        uint256 priceDecimals;
        address priceOracle;
        uint256 feeBps;
    }
    
    struct UserBalance {
        uint256 nativeBalance;
        mapping(address => uint256) tokenBalances;
    }
    
    // EntryPoint contract address (ERC-4337)
    address public entryPoint;
    
    // Supported tokens and their configurations
    mapping(address => TokenConfig) public supportedTokens;
    address[] public tokenList;
    
    // User balances
    mapping(address => UserBalance) public userBalances;
    
    // Gas price oracle
    uint256 public gasPriceCeiling;
    address public gasPriceOracle;
    
    // Fee collector
    address public feeCollector;
    uint256 public platformFeeBps;
    
    // ============ EVENTS ============
    event TokenSupported(
        address indexed token,
        uint256 conversionRate,
        uint256 feeBps
    );
    
    event TokenDeposited(
        address indexed user,
        address indexed token,
        uint256 amount
    );
    
    event TokenWithdrawn(
        address indexed user,
        address indexed token,
        uint256 amount
    );
    
    event UserOperationPaidWithToken(
        address indexed user,
        address indexed token,
        uint256 tokenAmount,
        uint256 ethCost
    );
    
    event ConversionRateUpdated(
        address indexed token,
        uint256 newRate
    );
    
    event GasPriceCeilingUpdated(uint256 newCeiling);

    // ============ MODIFIERS ============
    modifier onlyEntryPoint() {
        require(msg.sender == entryPoint, "TokenPaymaster: caller is not entry point");
        _;
    }
    
    modifier tokenSupported(address token) {
        require(supportedTokens[token].isSupported, "TokenPaymaster: token not supported");
        _;
    }

    // ============ CONSTRUCTOR ============
    constructor(
        address _entryPoint,
        address _feeCollector,
        uint256 _platformFeeBps
    ) {
        require(_entryPoint != address(0), "TokenPaymaster: invalid entry point");
        require(_feeCollector != address(0), "TokenPaymaster: invalid fee collector");
        require(_platformFeeBps <= MAX_BPS, "TokenPaymaster: fee too high");
        
        entryPoint = _entryPoint;
        feeCollector = _feeCollector;
        platformFeeBps = _platformFeeBps;
        gasPriceCeiling = 100 gwei;
    }

    // ============ EXTERNAL FUNCTIONS ============

    /**
     * @dev ERC-4337 validate UserOperation with token payment
     */
    function validatePaymasterUserOp(
        bytes calldata userOp,
        bytes32 userOpHash,
        uint256 maxCost
    ) external onlyEntryPoint returns (bytes memory context, uint256 validationData) {
        // Decode user operation
        (address sender, address paymaster, bytes memory paymasterData) = _decodeUserOp(userOp);
        
        require(paymaster == address(this), "TokenPaymaster: invalid paymaster");
        
        // Extract payment token and amount from paymaster data
        (address paymentToken, uint256 tokenAmount, bytes memory signature) = 
            abi.decode(paymasterData, (address, uint256, bytes));
        
        // Validate token payment
        _validateTokenPayment(sender, paymentToken, tokenAmount, maxCost, userOpHash, signature);
        
        // Deduct token balance
        userBalances[sender].tokenBalances[paymentToken] -= tokenAmount;
        
        // Take platform fee
        uint256 platformFee = tokenAmount.mul(platformFeeBps).div(MAX_BPS);
        if (platformFee > 0) {
            userBalances[sender].tokenBalances[paymentToken] -= platformFee;
            // Fee would be collected by fee collector
        }
        
        // Return context for post-operation
        context = abi.encode(sender, paymentToken, tokenAmount, maxCost);
        
        // Return validation data
        validationData = 0;
    }

    /**
     * @dev ERC-4337 post-operation callback
     */
    function postOp(
        bytes calldata context
    ) external onlyEntryPoint {
        // Context contains user and payment details
        // Additional logic can be implemented here if needed
        (address user, address paymentToken, uint256 tokenAmount, uint256 maxCost) = 
            abi.decode(context, (address, address, uint256, uint256));
        
        emit UserOperationPaidWithToken(user, paymentToken, tokenAmount, maxCost);
    }

    /**
     * @dev Deposit tokens for future gas payments
     */
    function depositTokens(
        address token,
        uint256 amount
    ) external nonReentrant tokenSupported(token) {
        require(amount > 0, "TokenPaymaster: zero amount");
        
        // Transfer tokens from user
        IERC20(token).safeTransferFrom(msg.sender, address(this), amount);
        
        // Update user balance
        userBalances[msg.sender].tokenBalances[token] += amount;
        
        emit TokenDeposited(msg.sender, token, amount);
    }

    /**
     * @dev Withdraw tokens from balance
     */
    function withdrawTokens(
        address token,
        uint256 amount
    ) external nonReentrant {
        require(amount > 0, "TokenPaymaster: zero amount");
        require(userBalances[msg.sender].tokenBalances[token] >= amount, "TokenPaymaster: insufficient balance");
        
        // Update balance
        userBalances[msg.sender].tokenBalances[token] -= amount;
        
        // Transfer tokens to user
        IERC20(token).safeTransfer(msg.sender, amount);
        
        emit TokenWithdrawn(msg.sender, token, amount);
    }

    /**
     * @dev Estimate token amount required for gas
     */
    function estimateTokenAmount(
        address token,
        uint256 ethAmount
    ) external view tokenSupported(token) returns (uint256) {
        TokenConfig memory config = supportedTokens[token];
        return ethAmount.mul(config.conversionRate).div(10 ** config.priceDecimals);
    }

    /**
     * @dev Get user token balance
     */
    function getUserTokenBalance(
        address user,
        address token
    ) external view returns (uint256) {
        return userBalances[user].tokenBalances[token];
    }

    // ============ ADMIN FUNCTIONS ============

    function supportToken(
        address token,
        uint256 conversionRate,
        uint256 priceDecimals,
        address priceOracle,
        uint256 feeBps
    ) external onlyOwner {
        require(token != address(0), "TokenPaymaster: invalid token");
        require(conversionRate > 0, "TokenPaymaster: invalid conversion rate");
        require(feeBps <= MAX_BPS, "TokenPaymaster: fee too high");
        
        if (!supportedTokens[token].isSupported) {
            tokenList.push(token);
        }
        
        supportedTokens[token] = TokenConfig({
            isSupported: true,
            conversionRate: conversionRate,
            priceDecimals: priceDecimals,
            priceOracle: priceOracle,
            feeBps: feeBps
        });
        
        emit TokenSupported(token, conversionRate, feeBps);
    }

    function updateConversionRate(
        address token,
        uint256 newRate
    ) external onlyOwner tokenSupported(token) {
        require(newRate > 0, "TokenPaymaster: invalid rate");
        supportedTokens[token].conversionRate = newRate;
        emit ConversionRateUpdated(token, newRate);
    }

    function updateGasPriceCeiling(uint256 newCeiling) external onlyOwner {
        gasPriceCeiling = newCeiling;
        emit GasPriceCeilingUpdated(newCeiling);
    }

    function updatePlatformFee(uint256 newFeeBps) external onlyOwner {
        require(newFeeBps <= MAX_BPS, "TokenPaymaster: fee too high");
        platformFeeBps = newFeeBps;
    }

    function updateFeeCollector(address newCollector) external onlyOwner {
        require(newCollector != address(0), "TokenPaymaster: invalid collector");
        feeCollector = newCollector;
    }

    function withdrawFees(address token) external onlyOwner {
        uint256 balance = IERC20(token).balanceOf(address(this));
        
        // Calculate total user balances to determine fee amount
        uint256 totalUserBalances = 0;
        for (uint256 i = 0; i < tokenList.length; i++) {
            // This would need to track total user balances per token
            // Simplified for illustration
        }
        
        uint256 feeAmount = balance - totalUserBalances;
        if (feeAmount > 0) {
            IERC20(token).safeTransfer(feeCollector, feeAmount);
        }
    }

    // ============ INTERNAL FUNCTIONS ============

    function _decodeUserOp(bytes calldata userOp) internal pure returns (
        address sender,
        address paymaster,
        bytes memory paymasterData
    ) {
        // Simplified UserOp decoding
        assembly {
            sender := calldataload(add(userOp.offset, 0x20))
            paymaster := calldataload(add(userOp.offset, 0x200))
            paymasterData.offset := add(userOp.offset, 0x240)
            paymasterData.length := calldataload(add(userOp.offset, 0x220))
        }
    }

    function _validateTokenPayment(
        address user,
        address paymentToken,
        uint256 tokenAmount,
        uint256 maxCost,
        bytes32 userOpHash,
        bytes memory signature
    ) internal view tokenSupported(paymentToken) {
        // Check user has sufficient token balance
        require(
            userBalances[user].tokenBalances[paymentToken] >= tokenAmount,
            "TokenPaymaster: insufficient token balance"
        );
        
        // Verify token amount covers gas cost
        TokenConfig memory config = supportedTokens[paymentToken];
        uint256 requiredTokens = maxCost.mul(config.conversionRate).div(10 ** config.priceDecimals);
        require(tokenAmount >= requiredTokens, "TokenPaymaster: insufficient token amount");
        
        // Verify user signature
        bytes32 hash = keccak256(abi.encode(userOpHash, paymentToken, tokenAmount));
        address recovered = _recoverSigner(hash, signature);
        require(recovered == user, "TokenPaymaster: invalid user signature");
    }

    function _recoverSigner(bytes32 hash, bytes memory signature) internal pure returns (address) {
        bytes32 ethSignedHash = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash));
        return _recover(ethSignedHash, signature);
    }

    function _recover(bytes32 hash, bytes memory signature) internal pure returns (address) {
        (bytes32 r, bytes32 s, uint8 v) = _splitSignature(signature);
        return ecrecover(hash, v, r, s);
    }

    function _splitSignature(bytes memory sig) internal pure returns (bytes32 r, bytes32 s, uint8 v) {
        require(sig.length == 65, "TokenPaymaster: invalid signature length");
        
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
        
        if (v < 27) v += 27;
        require(v == 27 || v == 28, "TokenPaymaster: invalid signature");
    }

    // ============ FALLBACK ============

    receive() external payable {}
}
