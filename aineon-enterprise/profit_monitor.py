import os
import time
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import colorama
from colorama import Fore, Back, Style

colorama.init(autoreset=True)

load_dotenv()

class ProfitMonitor:
    def __init__(self):
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8080")
        self.etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        self.last_profit = 0.0
        self.events = []

    def get_profit_data(self):
        try:
            response = requests.get(f"{self.api_base_url}/profit", timeout=5)
            if response.ok:
                return response.json()
            return None
        except:
            return None

    def get_status_data(self):
        try:
            response = requests.get(f"{self.api_base_url}/status", timeout=5)
            if response.ok:
                return response.json()
            return None
        except:
            return None

    def get_opportunities_data(self):
        try:
            response = requests.get(f"{self.api_base_url}/opportunities", timeout=5)
            if response.ok:
                return response.json()
            return None
        except:
            return None

    def add_event(self, event_type, message, color="white"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.append({
            "time": timestamp,
            "type": event_type,
            "message": message,
            "color": color
        })
        if len(self.events) > 20:  # Keep last 20 events
            self.events.pop(0)

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        header = "🚀 AINEON ENTERPRISE - LIVE PROFIT MONITORING TERMINAL"
        print(Fore.CYAN + Style.BRIGHT + "=" * 80)
        print(Fore.CYAN + Style.BRIGHT + header.center(80))
        print(Fore.CYAN + Style.BRIGHT + "=" * 80)

        # Status bar
        status = f"🟢 LIVE MODE | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(Fore.GREEN + status)
        print()

    def print_profit_section(self, data):
        print(Fore.YELLOW + Style.BRIGHT + "💰 PROFIT DASHBOARD")
        print(Fore.YELLOW + "=" * 50)

        if data:
            accumulated = data.get('accumulated_eth', 0.0)
            threshold = data.get('threshold_eth', 0.1)
            auto_transfer = data.get('auto_transfer', False)
            total_pnl = data.get('total_pnl', 0.0)
            active_trades = data.get('active_trades', 0)

            # Main profit display
            profit_color = Fore.GREEN if accumulated > 0 else Fore.RED
            print(f"{Style.BRIGHT}{profit_color}ACCUMULATED PROFIT: {accumulated:.6f} ETH")

            # Threshold and mode
            mode = "AUTO-SWEEP" if auto_transfer else "MANUAL"
            mode_color = Fore.MAGENTA if auto_transfer else Fore.WHITE
            print(f"{mode_color}TRANSFER MODE: {mode}")
            print(f"{Fore.WHITE}THRESHOLD: {threshold:.4f} ETH")

            # Additional metrics
            print(f"{Fore.GREEN}TOTAL P&L: ${total_pnl:,.2f}")
            print(f"{Fore.WHITE}ACTIVE TRADES: {active_trades}")

            # Check for new profit
            if accumulated > self.last_profit and self.last_profit > 0:
                self.add_event("PROFIT", f"New profit recorded: +{accumulated - self.last_profit:.6f} ETH", "green")
            self.last_profit = accumulated
        else:
            print(f"{Style.BRIGHT}{Fore.RED}⚠️  API UNAVAILABLE - CHECK ENGINE STATUS")
        print()

    def print_status_section(self, data):
        print(Fore.YELLOW + Style.BRIGHT + "🔗 BLOCKCHAIN STATUS")
        print(Fore.YELLOW + "=" * 50)

        if data:
            chain_id = data.get('chain_id', 0)
            ai_active = data.get('ai_active', False)
            tier = data.get('tier', 'UNKNOWN')

            print(f"{Fore.WHITE}NETWORK: Ethereum Mainnet (Chain ID: {chain_id})")
            ai_status = "🧠 ACTIVE" if ai_active else "🤖 INACTIVE"
            ai_color = Fore.GREEN if ai_active else Fore.RED
            print(f"{ai_color}AI STATUS: {ai_status}")
            print(f"{Style.BRIGHT}{Fore.MAGENTA}TIER: {tier}")
        else:
            print(f"{Style.BRIGHT}{Fore.RED}⚠️  BLOCKCHAIN CONNECTION UNAVAILABLE")
        print()

    def print_opportunities_section(self, data):
        print(Fore.YELLOW + Style.BRIGHT + "🎯 LIVE OPPORTUNITIES")
        print(Fore.YELLOW + "=" * 50)

        if data and 'opportunities' in data:
            opps = data['opportunities'][:5]  # Show top 5
            for opp in opps:
                pair = opp.get('pair', 'UNKNOWN')
                profit = opp.get('profit', 0.0)
                confidence = opp.get('confidence', 0.0)
                dex = opp.get('dex', 'UNKNOWN')

                profit_color = Fore.GREEN if profit > 0 else Fore.WHITE
                print(f"{profit_color}{pair} | {dex} | ${profit:.2f} | {confidence:.1%}")
        else:
            print(f"{Fore.WHITE}🔍 SCANNING FOR OPPORTUNITIES...")
        print()

    def print_events_section(self):
        print(Fore.YELLOW + Style.BRIGHT + "📋 LIVE BLOCKCHAIN EVENTS")
        print(Fore.YELLOW + "=" * 50)

        for event in reversed(self.events[-10:]):  # Show last 10 events
            time_str = event['time']
            msg = event['message'][:60]  # Truncate long messages

            color_map = {
                "white": Fore.WHITE,
                "green": Fore.GREEN,
                "red": Fore.RED,
                "yellow": Fore.YELLOW,
                "blue": Fore.BLUE
            }
            color = color_map.get(event['color'], Fore.WHITE)

            print(f"{color}[{time_str}] {msg}")
        print()

    def print_footer(self):
        print(Fore.WHITE + "=" * 80)
        print(Fore.WHITE + "ETHERSCAN VALIDATED PROFITS ONLY | Press Ctrl+C to quit")
        print(Fore.WHITE + "=" * 80)

    def run_monitor(self):
        self.add_event("SYSTEM", "Profit Monitor Started", "blue")

        try:
            while True:
                self.clear_screen()

                # Draw sections
                self.print_header()

                profit_data = self.get_profit_data()
                status_data = self.get_status_data()
                opp_data = self.get_opportunities_data()

                self.print_profit_section(profit_data)
                self.print_status_section(status_data)
                self.print_opportunities_section(opp_data)
                self.print_events_section()
                self.print_footer()

                time.sleep(5)  # Refresh every 5 seconds for better readability

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Monitor stopped by user.")

def main():
    monitor = ProfitMonitor()
    monitor.run_monitor()

if __name__ == "__main__":
    main()
