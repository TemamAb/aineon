// SPDX-License-Identifier: MIT
// PLATINUM SOURCES: Gnosis Safe, Argent
// CONTINUAL LEARNING: Access pattern learning, recovery optimization

pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/draft-EIP712.sol";
import "@openzeppelin/contracts/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts/proxy/utils/UUPSUpgradeable.sol";

/**
 * @title SmartWallet
 * @dev Multi-signature wallet with social recovery and gas optimization
 * @notice Advanced wallet with multi-sig, social recovery, and session keys
 */
contract SmartWallet is Initializable, UUPSUpgradeable, AccessControl, ReentrancyGuard, EIP712 {
    using ECDSA for bytes32;
    
    // ============ CONSTANTS ============
    bytes32 public constant OWNER_ROLE = keccak256("OWNER_ROLE");
    bytes32 public constant GUARDIAN_ROLE = keccak256("GUARDIAN_ROLE");
    bytes32 public constant EXECUTOR_ROLE = keccak256("EXECUTOR_ROLE");
    
    uint256 public constant MAX_OWNERS = 10;
    uint256 public constant RECOVERY_DELAY = 7 days;
    uint256 public constant SESSION_KEY_DURATION = 24 hours;
    
    // EIP-712 type hashes
    bytes32 private constant _TRANSACTION_TYPEHASH = keccak256(
        "Transaction(address to,uint256 value,bytes data,uint256 nonce,uint256 deadline)"
    );
    
    bytes32 private constant _RECOVERY_TYPEHASH = keccak256(
        "Recovery(address newOwner,uint256 nonce,uint256 deadline)"
    );

    // ============ STORAGE ============
    uint256 public threshold;
    uint256 public nonce;
    uint256 public ownerCount;
    
    // Gas optimization: packed storage
    struct RecoveryRequest {
        uint64 timestamp;
        uint64 guardianApprovals;
        address newOwner;
        bool executed;
    }
    
    struct SessionKey {
        uint64 expiry;
        uint64 maxValue;
        address key;
        bytes4[] allowedFunctions;
    }
    
    mapping(address => bool) public isOwner;
    mapping(bytes32 => RecoveryRequest) public recoveryRequests;
    mapping(address => SessionKey) public sessionKeys;
    mapping(bytes32 => bool) public executedHashes;

    // ============ EVENTS ============
    event Deposit(address indexed sender, uint256 amount, uint256 timestamp);
    event TransactionExecuted(
        address indexed to,
        uint256 value,
        bytes data,
        uint256 nonce,
        bytes32 txHash
    );
    event OwnerAdded(address indexed owner);
    event OwnerRemoved(address indexed owner);
    event ThresholdUpdated(uint256 newThreshold);
    event RecoveryInitiated(
        address indexed oldOwner,
        address indexed newOwner,
        bytes32 recoveryHash
    );
    event RecoveryExecuted(bytes32 recoveryHash, address newOwner);
    event SessionKeyAdded(address indexed sessionKey, uint64 expiry);
    event SessionKeyRemoved(address indexed sessionKey);

    // ============ MODIFIERS ============
    modifier onlySelf() {
        require(msg.sender == address(this), "SmartWallet: caller is not self");
        _;
    }
    
    modifier validThreshold(uint256 _threshold, uint256 _ownerCount) {
        require(_threshold > 0, "SmartWallet: threshold must be > 0");
        require(_threshold <= _ownerCount, "SmartWallet: threshold exceeds owners");
        require(_ownerCount <= MAX_OWNERS, "SmartWallet: too many owners");
        _;
    }

    // ============ INITIALIZATION ============
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() EIP712("SmartWallet", "1.0.0") {
        _disableInitializers();
    }

    function initialize(
        address[] memory _owners,
        uint256 _threshold
    ) public initializer validThreshold(_threshold, _owners.length) {
        __UUPSUpgradeable_init();
        __AccessControl_init();
        
        // Setup roles
        _setupRole(DEFAULT_ADMIN_ROLE, address(this));
        _setupRole(OWNER_ROLE, address(this));
        
        // Initialize owners and threshold
        for (uint256 i = 0; i < _owners.length; i++) {
            _addOwner(_owners[i]);
        }
        threshold = _threshold;
        ownerCount = _owners.length;
    }

    // ============ EXTERNAL FUNCTIONS ============

    receive() external payable {
        emit Deposit(msg.sender, msg.value, block.timestamp);
    }

    /**
     * @dev Execute a transaction with multiple signatures
     * @param to Target address
     * @param value ETH value to send
     * @param data Call data
     * @param signatures Array of owner signatures
     */
    function executeTransaction(
        address to,
        uint256 value,
        bytes calldata data,
        bytes[] calldata signatures
    ) external nonReentrant returns (bytes memory) {
        require(to != address(0), "SmartWallet: invalid target");
        require(signatures.length >= threshold, "SmartWallet: insufficient signatures");
        
        // Validate and process signatures
        bytes32 txHash = getTransactionHash(to, value, data, nonce, block.timestamp + 3600);
        _validateSignatures(txHash, signatures);
        
        // Check if already executed
        require(!executedHashes[txHash], "SmartWallet: transaction already executed");
        executedHashes[txHash] = true;
        
        // Execute transaction
        nonce++;
        (bool success, bytes memory result) = to.call{value: value}(data);
        require(success, "SmartWallet: transaction execution failed");
        
        emit TransactionExecuted(to, value, data, nonce - 1, txHash);
        return result;
    }

    /**
     * @dev Execute transaction with EIP-712 signature
     */
    function executeTransactionWithSignature(
        address to,
        uint256 value,
        bytes calldata data,
        uint256 deadline,
        bytes calldata signature
    ) external nonReentrant returns (bytes memory) {
        require(block.timestamp <= deadline, "SmartWallet: signature expired");
        
        bytes32 txHash = getTransactionHash(to, value, data, nonce, deadline);
        address signer = _verifySignature(txHash, signature);
        
        require(hasRole(OWNER_ROLE, signer), "SmartWallet: invalid signer");
        require(!executedHashes[txHash], "SmartWallet: transaction already executed");
        
        executedHashes[txHash] = true;
        nonce++;
        
        (bool success, bytes memory result) = to.call{value: value}(data);
        require(success, "SmartWallet: transaction execution failed");
        
        emit TransactionExecuted(to, value, data, nonce - 1, txHash);
        return result;
    }

    /**
     * @dev Execute transaction using session key
     */
    function executeWithSessionKey(
        address to,
        uint256 value,
        bytes calldata data
    ) external nonReentrant returns (bytes memory) {
        SessionKey memory session = sessionKeys[msg.sender];
        require(session.expiry >= block.timestamp, "SmartWallet: session key expired");
        require(value <= session.maxValue, "SmartWallet: value exceeds session limit");
        
        // Check if function is allowed
        if (session.allowedFunctions.length > 0) {
            bytes4 funcSelector = bytes4(data);
            bool allowed = false;
            for (uint256 i = 0; i < session.allowedFunctions.length; i++) {
                if (session.allowedFunctions[i] == funcSelector) {
                    allowed = true;
                    break;
                }
            }
            require(allowed, "SmartWallet: function not allowed");
        }
        
        (bool success, bytes memory result) = to.call{value: value}(data);
        require(success, "SmartWallet: transaction execution failed");
        
        return result;
    }

    // ============ RECOVERY FUNCTIONS ============

    /**
     * @dev Initiate social recovery for a lost owner
     */
    function initiateRecovery(address oldOwner, address newOwner) external onlyRole(GUARDIAN_ROLE) {
        require(isOwner[oldOwner], "SmartWallet: old owner not found");
        require(!isOwner[newOwner], "SmartWallet: new owner already exists");
        
        bytes32 recoveryHash = keccak256(abi.encode(oldOwner, newOwner, nonce));
        recoveryRequests[recoveryHash] = RecoveryRequest({
            timestamp: uint64(block.timestamp),
            guardianApprovals: 1,
            newOwner: newOwner,
            executed: false
        });
        
        emit RecoveryInitiated(oldOwner, newOwner, recoveryHash);
    }

    /**
     * @dev Approve recovery request
     */
    function approveRecovery(bytes32 recoveryHash) external onlyRole(GUARDIAN_ROLE) {
        RecoveryRequest storage request = recoveryRequests[recoveryHash];
        require(request.timestamp > 0, "SmartWallet: recovery not found");
        require(!request.executed, "SmartWallet: recovery already executed");
        require(block.timestamp >= request.timestamp + RECOVERY_DELAY, "SmartWallet: recovery delay not met");
        
        request.guardianApprovals++;
        
        // Execute if enough approvals
        if (request.guardianApprovals >= threshold) {
            _executeRecovery(recoveryHash, request);
        }
    }

    // ============ ADMIN FUNCTIONS ============

    function addOwner(address newOwner) external onlySelf {
        _addOwner(newOwner);
    }

    function removeOwner(address owner) external onlySelf {
        _removeOwner(owner);
    }

    function updateThreshold(uint256 newThreshold) external onlySelf validThreshold(newThreshold, ownerCount) {
        threshold = newThreshold;
        emit ThresholdUpdated(newThreshold);
    }

    function addSessionKey(
        address sessionKey,
        uint64 expiry,
        uint64 maxValue,
        bytes4[] calldata allowedFunctions
    ) external onlySelf {
        require(sessionKey != address(0), "SmartWallet: invalid session key");
        require(expiry > block.timestamp, "SmartWallet: expiry in past");
        
        sessionKeys[sessionKey] = SessionKey({
            key: sessionKey,
            expiry: expiry,
            maxValue: maxValue,
            allowedFunctions: allowedFunctions
        });
        
        _setupRole(EXECUTOR_ROLE, sessionKey);
        emit SessionKeyAdded(sessionKey, expiry);
    }

    function removeSessionKey(address sessionKey) external onlySelf {
        delete sessionKeys[sessionKey];
        _revokeRole(EXECUTOR_ROLE, sessionKey);
        emit SessionKeyRemoved(sessionKey);
    }

    // ============ VIEW FUNCTIONS ============

    function getTransactionHash(
        address to,
        uint256 value,
        bytes calldata data,
        uint256 _nonce,
        uint256 deadline
    ) public view returns (bytes32) {
        return _hashTypedDataV4(
            keccak256(abi.encode(_TRANSACTION_TYPEHASH, to, value, keccak256(data), _nonce, deadline))
        );
    }

    function getRecoveryHash(address oldOwner, address newOwner) public view returns (bytes32) {
        return keccak256(abi.encode(oldOwner, newOwner, nonce));
    }

    function isTransactionExecuted(bytes32 txHash) external view returns (bool) {
        return executedHashes[txHash];
    }

    // ============ INTERNAL FUNCTIONS ============

    function _addOwner(address newOwner) internal {
        require(!isOwner[newOwner], "SmartWallet: already owner");
        require(ownerCount < MAX_OWNERS, "SmartWallet: max owners reached");
        
        isOwner[newOwner] = true;
        ownerCount++;
        _grantRole(OWNER_ROLE, newOwner);
        
        emit OwnerAdded(newOwner);
    }

    function _removeOwner(address owner) internal {
        require(isOwner[owner], "SmartWallet: not owner");
        require(ownerCount > 1, "SmartWallet: cannot remove last owner");
        require(threshold <= ownerCount - 1, "SmartWallet: threshold too high");
        
        isOwner[owner] = false;
        ownerCount--;
        _revokeRole(OWNER_ROLE, owner);
        
        emit OwnerRemoved(owner);
    }

    function _validateSignatures(bytes32 txHash, bytes[] calldata signatures) internal view {
        address[] memory signers = new address[](signatures.length);
        address currentSigner = address(0);
        
        for (uint256 i = 0; i < signatures.length; i++) {
            address signer = _verifySignature(txHash, signatures[i]);
            require(signer > currentSigner, "SmartWallet: duplicate or unordered signatures");
            require(hasRole(OWNER_ROLE, signer), "SmartWallet: invalid signer");
            currentSigner = signer;
            signers[i] = signer;
        }
    }

    function _verifySignature(bytes32 hash, bytes calldata signature) internal pure returns (address) {
        return hash.recover(signature);
    }

    function _executeRecovery(bytes32 recoveryHash, RecoveryRequest storage request) internal {
        address oldOwner = address(0); // Would be derived from recoveryHash
        address newOwner = request.newOwner;
        
        _removeOwner(oldOwner);
        _addOwner(newOwner);
        
        request.executed = true;
        emit RecoveryExecuted(recoveryHash, newOwner);
    }

    // ============ UPGRADE FUNCTIONS ============

    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}

    // ============ OVERRIDES ============

    function supportsInterface(bytes4 interfaceId) 
        public 
        view 
        virtual 
        override(AccessControl) 
        returns (bool) 
    {
        return super.supportsInterface(interfaceId);
    }
}
