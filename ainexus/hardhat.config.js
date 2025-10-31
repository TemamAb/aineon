require("@nomiclabs/hardhat-ethers");
require("@nomiclabs/hardhat-waffle");

module.exports = {
  networks: {
    ethereum: {
      url: process.env.ETH_RPC_URL,
      accounts: {
        mnemonic: process.env.MNEMONIC, // Use mnemonic, not private key
        path: "m/44'/60'/0'/0",
        initialIndex: 0,
        count: 10
      }
    }
  }
};
