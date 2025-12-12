import os
import asyncio
from web3 import Web3

class ProfitManager:
    def __init__(self, w3: Web3, account_address: str, private_key: str):
        self.w3 = w3
        self.account_address = account_address
        self.private_key = private_key
        
        # Defaults
        self.accumulated_profit_eth = 0.0
        self.threshold_eth = 0.01  # Default threshold
        self.auto_transfer_enabled = True # Enabled for DEMO
        self.target_wallet = os.getenv("PROFIT_WALLET", account_address) # Default to self if not set

    def update_config(self, enabled: bool, threshold: float, target_wallet: str = None):
        self.auto_transfer_enabled = enabled
        self.threshold_eth = float(threshold)
        if target_wallet:
            self.target_wallet = target_wallet
        print(f"[PROFIT] Config Updated: Auto={self.auto_transfer_enabled}, Threshold={self.threshold_eth} ETH")

    async def record_profit(self, amount_eth: float):
        """Records profit from a successful trade and checks threshold"""
        self.accumulated_profit_eth += amount_eth
        print(f"[PROFIT] +{amount_eth:.4f} ETH | Total: {self.accumulated_profit_eth:.4f} ETH")
        
        if self.auto_transfer_enabled:
            await self.check_and_transfer()

    async def check_and_transfer(self):
        """Checks if profit exceeds threshold and transfers if so"""
        if self.accumulated_profit_eth >= self.threshold_eth:
            amount_to_send = self.accumulated_profit_eth
            print(f"[PROFIT] Threshold met ({self.accumulated_profit_eth} >= {self.threshold_eth}). Initiating Transfer...")
            
            success = await self._execute_transfer(amount_to_send)
            if success:
                self.accumulated_profit_eth = 0.0
                print(f"[PROFIT] Transfer Complete. Resetting accumulator.")

    async def _execute_transfer(self, amount_eth: float):
        """Executes the actual ETH transfer"""
        try:
            print(f">>> SENDING {amount_eth:.4f} ETH to {self.target_wallet} <<<")
            
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
            
            print(f">>> TRANSFER SUCCESS: {self.w3.to_hex(tx_hash)}")
            return True
        except Exception as e:
            print(f"[PROFIT] Transfer Failed (Real Mode): {e}")
            return False

    def get_stats(self):
        return {
            "accumulated_eth": self.accumulated_profit_eth,
            "threshold_eth": self.threshold_eth,
            "auto_transfer_enabled": self.auto_transfer_enabled,
            "target_wallet": self.target_wallet
        }
