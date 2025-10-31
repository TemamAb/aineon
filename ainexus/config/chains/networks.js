module.exports = {
    ethereum: {
        chainId: 1,
        rpc: process.env.ETH_RPC_URL,
        explorer: 'https://etherscan.io',
        flashLoanProviders: ['AAVE', 'DYDX']
    },
    polygon: {
        chainId: 137,
        rpc: process.env.POLYGON_RPC_URL,
        explorer: 'https://polygonscan.com',
        flashLoanProviders: ['AAVE', 'QUICKSWAP']
    },
    arbitrum: {
        chainId: 42161,
        rpc: process.env.ARBITRUM_RPC_URL,
        explorer: 'https://arbiscan.io',
        flashLoanProviders: ['AAVE']
    }
};
