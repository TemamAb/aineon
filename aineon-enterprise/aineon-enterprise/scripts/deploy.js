const hre = require("hardhat");

async function main() {
  const AavePoolProvider = "0x2f39d218133AFa93a1285242190238366001E95u"; // Mainnet
  
  console.log("Deploying AineonUltra...");
  const aineon = await hre.ethers.deployContract("AineonUltra", [AavePoolProvider]);
  await aineon.waitForDeployment();

  console.log("AineonUltra Deployed to:", aineon.target);
  console.log(">> ACTION: Copy this address to your .env file as CONTRACT_ADDRESS");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
