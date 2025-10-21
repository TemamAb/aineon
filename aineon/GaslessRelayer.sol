// SPDX-License-Identifier: MIT
// PLATINUM SOURCES: OpenGSN, Biconomy
// CONTINUAL LEARNING: Relay efficiency learning, spam prevention

pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/draft-EIP712.sol";

/**
 * @title GaslessRelayer
 * @dev ERC-2771 compatible meta-transaction relayer with spam protection
 * @notice Efficient gasless transaction relay with nonce management and fee optimization
 */
contract GaslessRelayer is Ownable, ReentrancyGuard, EIP712 {
    using ECDSA for bytes32;
    
    // ============ CONSTANTS ============
    bytes32 private constant _META_TX_TYPEHASH = keccak256(
        "MetaTransaction(address from,address to,uint256 value,bytes data,uint256 nonce,uint256 deadline)"
    );
    
    uint256 public constant MAX_FEE_BPS = 500; // 5% maximum fee
    uint256 public constant MIN_FEE = 0.001 ether;

    // ============ STORAGE ============
    struct RelayConfig {
        uint256 feeBps;
        uint256 minFee;
        uint256 maxFee;
        bool isActive;
    }
    
    // Trusted forwarder for ERC-2771
    address public trustedForwarder;
    
    // Relay configuration
    RelayConfig public relayConfig;
    
    // Nonce management for replay protection
    mapping(address => uint256) public nonces;
    
    // Fee collector address
    address public feeCollector;
    
    // ============ EVENTS ============
    event MetaTransactionExecuted(
        address indexed from,
        address indexed to,
        uint256 value,
        bytes data,
        uint256 nonce,
        uint256 fee
    );
    
    event TrustedForwarderUpdated(address indexed newForwarder);
    event RelayConfigUpdated(uint256 feeBps, uint256 minFee, uint256 maxFee);
    event FeeCollectorUpdated(address indexed newCollector);
    event FeesWithdrawn(address indexed collector, uint256 amount);

    // ============ MODIFIERS ============
    modifier onlyTrustedForwarder() {
        require(msg.sender == trustedForwarder, "GaslessRelayer: unauthorized forwarder");
        _;
    }

    // ============ CONSTRUCTOR ============
    constructor(
        address _trustedForwarder,
        address _feeCollector
    ) EIP712("GaslessRelayer", "1.0.0") {
        require(_trustedForwarder != address(0), "GaslessRelayer: invalid forwarder");
        require(_feeCollector != address(0), "GaslessRelayer: invalid fee collector");
        
        trustedForwarder = _trustedForwarder;
        feeCollector = _feeCollector;
        
        // Default relay configuration
        relayConfig = RelayConfig({
            feeBps: 100, // 1% fee
            minFee: MIN_FEE,
            maxFee: 0.01 ether,
            isActive: true
        });
    }

    // ============ EXTERNAL FUNCTIONS ============

    /**
     * @dev Execute meta-transaction with EIP-712 signature
     */
    function executeMetaTransaction(
        address from,
        address to,
        uint256 value,
        bytes calldata data,
        uint256 deadline,
        bytes calldata signature
    ) external nonReentrant returns (bytes memory) {
        require(relayConfig.isActive, "GaslessRelayer: relay inactive");
        require(block.timestamp <= deadline, "GaslessRelayer: signature expired");
        
        // Verify signature and nonce
        uint256 currentNonce = nonces[from];
        bytes32 structHash = keccak256(
            abi.encode(_META_TX_TYPEHASH, from, to, value, keccak256(data), currentNonce, deadline)
        );
        bytes32 hash = _hashTypedDataV4(structHash);
        
        address signer = hash.recover(signature);
        require(signer == from, "GaslessRelayer: invalid signature");
        
        // Update nonce
        nonces[from] = currentNonce + 1;
        
        // Calculate and collect fee
        uint256 fee = _calculateFee(value);
        require(msg.value >= fee, "GaslessRelayer: insufficient fee");
        
        // Transfer fee to collector
        if (fee > 0) {
            payable(feeCollector).transfer(fee);
        }
        
        // Refund excess fee
        if (msg.value > fee) {
            payable(msg.sender).transfer(msg.value - fee);
        }
        
        // Execute the actual transaction
        (bool success, bytes memory result) = to.call{value: value}(data);
        require(success, "GaslessRelayer: transaction execution failed");
        
        emit MetaTransactionExecuted(from, to, value, data, currentNonce, fee);
        return result;
    }

    /**
     * @dev ERC-2771 forwarder verification
     */
    function isTrustedForwarder(address forwarder) external view returns (bool) {
        return forwarder == trustedForwarder;
    }

    /**
     * @dev Get current nonce for an address
     */
    function getNonce(address from) external view returns (uint256) {
        return nonces[from];
    }

    /**
     * @dev Calculate fee for a transaction
     */
    function calculateFee(uint256 value) external view returns (uint256) {
        return _calculateFee(value);
    }

    /**
     * @dev Get meta-transaction hash
     */
    function getMetaTransactionHash(
        address from,
        address to,
        uint256 value,
        bytes calldata data,
        uint256 nonce,
        uint256 deadline
    ) external view returns (bytes32) {
        bytes32 structHash = keccak256(
            abi.encode(_META_TX_TYPEHASH, from, to, value, keccak256(data), nonce, deadline)
        );
        return _hashTypedDataV4(structHash);
    }

    // ============ ADMIN FUNCTIONS ============

    function updateTrustedForwarder(address newForwarder) external onlyOwner {
        require(newForwarder != address(0), "GaslessRelayer: invalid forwarder");
        trustedForwarder = newForwarder;
        emit TrustedForwarderUpdated(newForwarder);
    }

    function updateRelayConfig(
        uint256 feeBps,
        uint256 minFee,
        uint256 maxFee
    ) external onlyOwner {
        require(feeBps <= MAX_FEE_BPS, "GaslessRelayer: fee too high");
        require(minFee <= maxFee, "GaslessRelayer: invalid fee range");
        
        relayConfig.feeBps = feeBps;
        relayConfig.minFee = minFee;
        relayConfig.maxFee = maxFee;
        
        emit RelayConfigUpdated(feeBps, minFee, maxFee);
    }

    function setRelayActive(bool active) external onlyOwner {
        relayConfig.isActive = active;
    }

    function updateFeeCollector(address newCollector) external onlyOwner {
        require(newCollector != address(0), "GaslessRelayer: invalid collector");
        feeCollector = newCollector;
        emit FeeCollectorUpdated(newCollector);
    }

    function withdrawFees() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "GaslessRelayer: no fees to withdraw");
        
        payable(feeCollector).transfer(balance);
        emit FeesWithdrawn(feeCollector, balance);
    }

    // ============ INTERNAL FUNCTIONS ============

    function _calculateFee(uint256 value) internal view returns (uint256) {
        uint256 calculatedFee = (value * relayConfig.feeBps) / 10000;
        
        if (calculatedFee < relayConfig.minFee) {
            return relayConfig.minFee;
        }
        
        if (calculatedFee > relayConfig.maxFee) {
            return relayConfig.maxFee;
        }
        
        return calculatedFee;
    }

    // ============ FALLBACK ============

    receive() external payable {}
}
