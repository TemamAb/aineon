const hre = require("hardhat");
async function main() {
  console.log("íº€ Deploying AINEON...");
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deployer:", deployer.address);
  
  const MultiSig = await hre.ethers.getContractFactory("MultiSig");
  const multiSig = await MultiSig.deploy([deployer.address], 1);
  await multiSig.deployed();
  console.log("âœ… MultiSig deployed:", multiSig.address);
  console.log("í¾¯ Ready for mainnet deployment!");
}
main().catch(console.error);
