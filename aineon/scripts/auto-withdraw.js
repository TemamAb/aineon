const { ethers } = require('ethers');

class ProfitAutoWithdrawal {
    constructor() {
        this.threshold = 1000; // $1,000
        this.interval = 300000; // 5 minutes
        this.bossWallet = process.env.BOSS_WALLET_ADDRESS;
    }

    async checkAndWithdraw() {
        const currentProfits = await this.getCurrentProfits();
        console.log(`í²° Current profits: $${currentProfits}`);
        
        if (currentProfits > this.threshold) {
            console.log(`íº€ Withdrawing $${currentProfits} to Boss wallet...`);
            await this.executeWithdrawal(currentProfits);
        }
    }

    async getCurrentProfits() {
        // Connect to your profit tracking system
        // This would interface with your trading engine
        return 116137; // Current profit amount
    }

    async executeWithdrawal(amount) {
        try {
            // Connect to MetaMask/provider
            const provider = new ethers.providers.Web3Provider(window.ethereum);
            const signer = provider.getSigner();
            
            // Create transaction (simplified)
            const tx = await signer.sendTransaction({
                to: this.bossWallet,
                value: ethers.utils.parseEther(this.convertToETH(amount)),
                gasLimit: 21000
            });
            
            console.log(`âœ… Withdrawal sent: ${tx.hash}`);
            return tx;
            
        } catch (error) {
            console.error('âŒ Withdrawal failed:', error);
        }
    }

    convertToETH(usdAmount) {
        // Simple conversion - in production, use real exchange rate
        return (usdAmount / 2000).toFixed(4); // Assuming $2000/ETH
    }

    startMonitoring() {
        console.log('í´ Starting profit withdrawal monitoring...');
        setInterval(() => this.checkAndWithdraw(), this.interval);
        this.checkAndWithdraw(); // Immediate check
    }
}

// Start the system
const withdrawalSystem = new ProfitAutoWithdrawal();
withdrawalSystem.startMonitoring();
