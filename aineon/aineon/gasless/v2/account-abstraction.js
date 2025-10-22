// SMART CONTRACT WALLETS & ACCOUNT ABSTRACTION
class AccountAbstraction {
  constructor() {
    this.walletFactory = process.env.WALLET_FACTORY;
    this.entryPoint = process.env.ENTRY_POINT;
  }

  async createSmartWallet(owner, salt = 0) {
    const initCode = this.encodeWalletCreation(owner, salt);
    const walletAddress = await this.calculateWalletAddress(owner, salt);
    
    return {
      address: walletAddress,
      initCode: initCode,
      owner: owner,
      salt: salt,
      isDeployed: await this.isWalletDeployed(walletAddress)
    };
  }

  encodeWalletCreation(owner, salt) {
    // Encode factory.createWallet(owner, salt)
    return ethers.utils.hexConcat([
      this.walletFactory,
      ethers.utils.hexDataSlice(
        new ethers.utils.Interface([
          'function createWallet(address owner, uint256 salt) returns (address)'
        ]).encodeFunctionData('createWallet', [owner, salt]),
      0
    ]);
  }

  async calculateWalletAddress(owner, salt) {
    // Calculate counterfactual address
    const initCode = this.encodeWalletCreation(owner, salt);
    const initCodeHash = ethers.utils.keccak256(initCode);
    return ethers.utils.getCreate2Address(
      this.walletFactory,
      ethers.utils.hexZeroPad(ethers.utils.hexlify(salt), 32),
      initCodeHash
    );
  }

  async executeBatchFromSmartWallet(walletAddress, transactions, signature) {
    const userOperation = {
      sender: walletAddress,
      nonce: await this.getNonce(walletAddress),
      callData: this.encodeExecuteBatch(transactions),
      signature: signature,
      paymasterAndData: await this.getPaymasterData()
    };

    return await this.submitUserOperation(userOperation);
  }

  encodeExecuteBatch(transactions) {
    // Encode wallet.executeBatch(txs)
    const iface = new ethers.utils.Interface([
      'function executeBatch(address[] targets, uint256[] values, bytes[] calldatas)'
    ]);
    
    const targets = transactions.map(tx => tx.to);
    const values = transactions.map(tx => tx.value || 0);
    const calldatas = transactions.map(tx => tx.data || '0x');
    
    return iface.encodeFunctionData('executeBatch', [targets, values, calldatas]);
  }
}
module.exports = AccountAbstraction;
