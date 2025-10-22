// SPDX-License-Identifier: MIT
// PLATINUM SOURCES: ERC-4337, Stackup
// CONTINUAL LEARNING: Sponsorship strategy learning, cost optimization

pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/proxy/utils/Initializable.sol";

/**
 * @title Paymaster
 * @dev ERC-4337 Paymaster for gasless transaction sponsorship
 * @notice Advanced paymaster with whitelisting, spending limits, and gas optimization
 */
contract Paymaster is Initializable, Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;
    using ECDSA for bytes32;
    
    // ============ CONSTANTS ============
    uint256 public constant MAX_BPS = 10_000;
    uint256 public constant MIN_STAKE = 0.1 ether;
    
    // ============ STORAGE ============
    struct SponsorConfig {
        uint256 balance;
        uint256 maxSpendPerUser;
        uint256 maxSpendPerTx;
        uint256 userCount;
        bool isActive;
    }
    
    struct UserSpending {
        uint256 totalSpent;
        uint256 lastTxTime;
        uint256 txCount;
    }
    
    // EntryPoint contract address (ERC-4337)
    address public entryPoint;
    
    // Sponsor configurations
    mapping(address => SponsorConfig) public sponsors;
    mapping(address => mapping(address => UserSpending)) public userSpending;
    mapping(address => bool) public whitelistedContracts;
    
    // Gas price oracle for cost calculations
    uint256 public gasPriceCeiling;
    
    // ============ EVENTS ============
    event SponsorRegistered(address indexed sponsor, uint256 initialBalance);
    event SponsorDeposited(address indexed sponsor, uint256 amount);
    event UserOperationSponsored(
        address indexed sponsor,
        address indexed user,
        uint256 actualGasCost,
        uint256 actualUserOpCost
    );
    event GasPriceCeilingUpdated(uint256 newCeiling);
    event ContractWhitelisted(address indexed contractAddress);
    event ContractBlacklisted(address indexed contractAddress);

    // ============ MODIFIERS ============
    modifier onlyEntryPoint() {
        require(msg.sender == entryPoint, "Paymaster: caller is not entry point");
        _;
    }
    
    modifier onlyActiveSponsor(address sponsor) {
        require(sponsors[sponsor].isActive, "Paymaster: sponsor not active");
        _;
    }

    // ============ INITIALIZATION ============
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(address _entryPoint, address initialOwner) public initializer {
        require(_entryPoint != address(0), "Paymaster: invalid entry point");
        _transferOwnership(initialOwner);
        
        entryPoint = _entryPoint;
        gasPriceCeiling = 100 gwei; // Initial gas price ceiling
    }

    // ============ EXTERNAL FUNCTIONS ============

    /**
     * @dev ERC-4337 validate UserOperation
     */
    function validatePaymasterUserOp(
        bytes calldata userOp,
        bytes32 userOpHash,
        uint256 maxCost
    ) external onlyEntryPoint returns (bytes memory context, uint256 validationData) {
        // Decode user operation
        (address sender, address paymaster, bytes memory paymasterData) = _decodeUserOp(userOp);
        
        require(paymaster == address(this), "Paymaster: invalid paymaster");
        
        // Extract sponsor from paymaster data
        (address sponsor, bytes memory signature) = abi.decode(paymasterData, (address, bytes));
        
        // Validate sponsor and signature
        _validateSponsorship(sponsor, sender, userOpHash, signature, maxCost);
        
        // Update spending tracking
        _updateUserSpending(sponsor, sender, maxCost);
        
        // Return context for post-operation
        context = abi.encode(sponsor, sender, maxCost);
        
        // Return validation data (always valid for now)
        validationData = 0;
    }

    /**
     * @dev ERC-4337 post-operation callback
     */
    function postOp(
        bytes calldata context,
        uint256 actualGasCost
    ) external onlyEntryPoint {
        (address sponsor, address user, uint256 maxCost) = abi.decode(context, (address, address, uint256));
        
        // Deduct actual gas cost from sponsor balance
        sponsors[sponsor].balance -= actualGasCost;
        
        emit UserOperationSponsored(sponsor, user, actualGasCost, maxCost);
    }

    /**
     * @dev Register as a sponsor with initial deposit
     */
    function registerSponsor(
        uint256 maxSpendPerUser,
        uint256 maxSpendPerTx
    ) external payable nonReentrant {
        require(msg.value >= MIN_STAKE, "Paymaster: insufficient stake");
        require(!sponsors[msg.sender].isActive, "Paymaster: already registered");
        
        sponsors[msg.sender] = SponsorConfig({
            balance: msg.value,
            maxSpendPerUser: maxSpendPerUser,
            maxSpendPerTx: maxSpendPerTx,
            userCount: 0,
            isActive: true
        });
        
        emit SponsorRegistered(msg.sender, msg.value);
    }

    /**
     * @dev Deposit more funds as a sponsor
     */
    function deposit() external payable onlyActiveSponsor(msg.sender) {
        sponsors[msg.sender].balance += msg.value;
        emit SponsorDeposited(msg.sender, msg.value);
    }

    /**
     * @dev Withdraw sponsor funds
     */
    function withdraw(uint256 amount) external nonReentrant onlyActiveSponsor(msg.sender) {
        require(amount <= sponsors[msg.sender].balance, "Paymaster: insufficient balance");
        
        sponsors[msg.sender].balance -= amount;
        payable(msg.sender).transfer(amount);
    }

    // ============ ADMIN FUNCTIONS ============

    function updateGasPriceCeiling(uint256 newCeiling) external onlyOwner {
        gasPriceCeiling = newCeiling;
        emit GasPriceCeilingUpdated(newCeiling);
    }

    function whitelistContract(address contractAddress) external onlyOwner {
        whitelistedContracts[contractAddress] = true;
        emit ContractWhitelisted(contractAddress);
    }

    function blacklistContract(address contractAddress) external onlyOwner {
        whitelistedContracts[contractAddress] = false;
        emit ContractBlacklisted(contractAddress);
    }

    function setEntryPoint(address newEntryPoint) external onlyOwner {
        require(newEntryPoint != address(0), "Paymaster: invalid entry point");
        entryPoint = newEntryPoint;
    }

    // ============ VIEW FUNCTIONS ============

    function getSponsorBalance(address sponsor) external view returns (uint256) {
        return sponsors[sponsor].balance;
    }

    function canSponsor(
        address sponsor,
        address user,
        uint256 maxCost
    ) external view returns (bool) {
        if (!sponsors[sponsor].isActive) return false;
        if (sponsors[sponsor].balance < maxCost) return false;
        if (maxCost > sponsors[sponsor].maxSpendPerTx) return false;
        
        UserSpending memory spending = userSpending[sponsor][user];
        if (spending.totalSpent + maxCost > sponsors[sponsor].maxSpendPerUser) return false;
        
        return true;
    }

    function estimateMaxCost(
        uint256 callGasLimit,
        uint256 verificationGasLimit,
        uint256 preVerificationGas
    ) external view returns (uint256) {
        uint256 estimatedGas = callGasLimit + verificationGasLimit + preVerificationGas;
        return estimatedGas * gasPriceCeiling;
    }

    // ============ INTERNAL FUNCTIONS ============

    function _decodeUserOp(bytes calldata userOp) internal pure returns (
        address sender,
        address paymaster,
        bytes memory paymasterData
    ) {
        // Simplified UserOp decoding - would use proper ABI decoding in production
        // UserOp structure: [sender, nonce, initCode, callData, callGasLimit, verificationGasLimit, preVerificationGas, maxFeePerGas, maxPriorityFeePerGas, paymaster, paymasterData, signature]
        assembly {
            sender := calldataload(add(userOp.offset, 0x20))
            paymaster := calldataload(add(userOp.offset, 0x200))
            paymasterData.offset := add(userOp.offset, 0x240)
            paymasterData.length := calldataload(add(userOp.offset, 0x220))
        }
    }

    function _validateSponsorship(
        address sponsor,
        address user,
        bytes32 userOpHash,
        bytes memory signature,
        uint256 maxCost
    ) internal view {
        // Check sponsor is active and has sufficient balance
        require(sponsors[sponsor].isActive, "Paymaster: sponsor not active");
        require(sponsors[sponsor].balance >= maxCost, "Paymaster: insufficient sponsor balance");
        
        // Check spending limits
        require(maxCost <= sponsors[sponsor].maxSpendPerTx, "Paymaster: exceeds max spend per tx");
        
        UserSpending memory spending = userSpending[sponsor][user];
        require(
            spending.totalSpent + maxCost <= sponsors[sponsor].maxSpendPerUser,
            "Paymaster: exceeds max spend per user"
        );
        
        // Verify sponsor signature
        bytes32 hash = keccak256(abi.encode(userOpHash, sponsor, maxCost));
        address recovered = hash.recover(signature);
        require(recovered == sponsor, "Paymaster: invalid sponsor signature");
    }

    function _updateUserSpending(address sponsor, address user, uint256 cost) internal {
        UserSpending storage spending = userSpending[sponsor][user];
        
        if (spending.txCount == 0) {
            sponsors[sponsor].userCount++;
        }
        
        spending.totalSpent += cost;
        spending.lastTxTime = block.timestamp;
        spending.txCount++;
    }

    // ============ FALLBACK ============

    receive() external payable {}
}
