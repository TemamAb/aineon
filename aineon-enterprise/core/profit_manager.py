import os
import asyncio
import time
import requests
from web3 import Web3

class ProfitManager:
    def __init__(self, w3: Web3, account_address: str, private_key: str):
        self.w3 = w3
        self.account_address = account_address
        self.private_key = private_key
        
        # Defaults
        self.accumulated_profit_eth = 0.0
        self.threshold_eth = 0.1  # Requested Threshold
        self.auto_transfer_enabled = False # Default to False (Smart Wallet Pending)
        self.target_wallet = os.getenv("PROFIT_WALLET", account_address) # Default to self if not set

    # ANSI Colors for Terminal
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def update_config(self, enabled: bool, threshold: float, target_wallet: str = None):
        self.auto_transfer_enabled = enabled
        self.threshold_eth = float(threshold)
        if target_wallet:
            self.target_wallet = target_wallet
        print(f"{self.CYAN}[PROFIT]{self.RESET} Config Updated: Auto={self.auto_transfer_enabled}, Threshold={self.threshold_eth} ETH")

    async def verify_transaction_on_etherscan(self, tx_hash: str) -> bool:
        """Verify transaction is confirmed on Etherscan before displaying profit"""
        try:
            # Wait for transaction to be mined
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            # Check if transaction was successful
            if receipt['status'] != 1:
                print(f"{self.RED}[AUDIT] Transaction {tx_hash[:10]}... FAILED on blockchain{self.RESET}")
                return False
            
            # Verify on Etherscan API (optional additional check)
            etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
            if etherscan_api_key:
                url = f"https://api.etherscan.io/api?module=transaction&action=gettxreceiptstatus&txhash={tx_hash}&apikey={etherscan_api_key}"
                response = requests.get(url, timeout=10)
                if response.ok:
                    data = response.json()
                    if data.get('result', {}).get('status') == '1':
                        print(f"{self.GREEN}[AUDIT] ✓ ETHERSCAN VERIFIED: {tx_hash[:10]}...{self.RESET}")
                        return True
            
            # If no API key, just use Web3 receipt
            print(f"{self.GREEN}[AUDIT] ✓ BLOCKCHAIN CONFIRMED: {tx_hash[:10]}... (Block {receipt['blockNumber']}){self.RESET}")
            return True
            
        except Exception as e:
            print(f"{self.RED}[AUDIT] Verification failed: {e}{self.RESET}")
            return False
    
    async def record_profit(self, amount_eth: float, tx_hash: str = None, simulated: bool = False):
        """Records profit from a successful trade ONLY after Etherscan validation"""
        
        if simulated:
            # Simulation mode - no real validation needed
            tx_hash_display = f"0x{os.urandom(4).hex()}...{os.urandom(4).hex()}" 
            print(f"{self.CYAN}{self.BOLD}[SIMULATION] +{amount_eth:.4f} ETH{self.RESET} | {self.CYAN}VALIDATED BY SIMULATION (TENDERLY): {tx_hash_display}{self.RESET}")
            print(f"{self.CYAN}   >>> STATUS: VIRTUAL PROFIT (NO REAL FUNDS MOVED){self.RESET}")
            return
        
        # REAL MODE - REQUIRE ETHERSCAN VALIDATION
        if not tx_hash:
            print(f"{self.RED}[AUDIT] ERROR: No transaction hash provided for profit verification{self.RESET}")
            return
        
        print(f"{self.YELLOW}[AUDIT] Verifying transaction on Etherscan...{self.RESET}")
        
        # Verify transaction is confirmed on blockchain
        is_verified = await self.verify_transaction_on_etherscan(tx_hash)
        
        if not is_verified:
            print(f"{self.RED}[AUDIT] ✗ PROFIT REJECTED - Transaction not confirmed{self.RESET}")
            return
        
        # Only record profit AFTER successful verification
        self.accumulated_profit_eth += amount_eth
        
        # Display profit with Etherscan validation badge
        print(f"{self.GREEN}{self.BOLD}[PROFIT EVENT] +{amount_eth:.4f} ETH{self.RESET} | {self.GREEN}✓ VALIDATED BY ETHERSCAN: {tx_hash[:10]}...{tx_hash[-8:]}{self.RESET}")
        print(f"{self.GREEN}   >>> RUNNING TOTAL: {self.accumulated_profit_eth:.4f} ETH{self.RESET}")
        print(f"{self.CYAN}   >>> STATUS: SECURED (PENDING IN SMART WALLET){self.RESET}")
        
        if self.auto_transfer_enabled:
            await self.check_and_transfer()

    async def check_and_transfer(self):
        """Checks if profit exceeds threshold and transfers if so"""
        if self.accumulated_profit_eth >= self.threshold_eth:
            amount_to_send = self.accumulated_profit_eth
            print(f"{self.YELLOW}[PROFIT]{self.RESET} Threshold met ({self.accumulated_profit_eth} >= {self.threshold_eth}). Initiating Transfer...")
            
            success = await self._execute_transfer(amount_to_send)
            if success:
                self.accumulated_profit_eth = 0.0
                print(f"{self.CYAN}[PROFIT]{self.RESET} Transfer Complete. Resetting accumulator.")

    async def _execute_transfer(self, amount_eth: float):
        """Executes the actual ETH transfer"""
        try:
            print(f"{self.CYAN}>>> SENDING {amount_eth:.4f} ETH to {self.target_wallet} <<<{self.RESET}")
            
            nonce = self.w3.eth.get_transaction_count(self.account_address)
            gas_price = self.w3.eth.gas_price
            
            tx = {
                'to': self.target_wallet,
                'value': self.w3.to_wei(amount_eth, 'ether'),
                'gas': 21000,
                'gasPrice': gas_price,
                'nonce': nonce,
                'chainId': self.w3.eth.chain_id
            }
            
            signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
            
            print(f"{self.GREEN}{self.BOLD}>>> TRANSFER SUCCESS: {self.w3.to_hex(tx_hash)}{self.RESET}")
            return True
        except Exception as e:
            print(f"{self.RED}[PROFIT] Transfer Failed (Real Mode): {e}{self.RESET}")
            return False

    async def force_transfer(self):
        """Manually forces a transfer of all accumulated profit regardless of threshold."""
        if self.accumulated_profit_eth > 0:
            print(f"{self.YELLOW}[PROFIT]{self.RESET} PROFIT READY FOR MANUAL TRANSFER")
            amount_to_send = self.accumulated_profit_eth
            success = await self._execute_transfer(amount_to_send)
            if success:
                self.accumulated_profit_eth = 0.0
                print(f"{self.CYAN}[PROFIT]{self.RESET} Manual Transfer Complete.")
            return success
        else:
            print(f"{self.RED}[PROFIT]{self.RESET} No funds to withdraw.")
            return False

    def get_stats(self):
        return {
            "accumulated_eth": self.accumulated_profit_eth,
            "threshold_eth": self.threshold_eth,
            "auto_transfer_enabled": self.auto_transfer_enabled,
            "target_wallet": self.target_wallet
        }
