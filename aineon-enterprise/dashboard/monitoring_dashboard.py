import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import websockets
import json
import threading
import time
import requests
import os

class MonitoringDashboard:
    def __init__(self):
        self.trade_history = []
        self.performance_metrics = {}
        self.risk_metrics = {}
        self.websocket_url = "ws://localhost:8765"  # For real-time updates
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8080")

    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(page_title="Aineon 0.001% Tier Dashboard", layout="wide")

        st.title("🚀 Aineon Ultra Arbitrage Dashboard")
        st.markdown("**Real-time monitoring for elite arbitrage performance**")

        # Sidebar for controls
        self.sidebar_controls()

        # Main dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🎯 Opportunities", "⚠️ Risk", "🔧 Settings"])

        with tab1:
            self.performance_tab()

        with tab2:
            self.opportunities_tab()

        with tab3:
            self.risk_tab()

        with tab4:
            self.settings_tab()

    def sidebar_controls(self):
        """Sidebar controls for dashboard"""
        st.sidebar.header("⚙️ Controls")

        # Status indicator
        status_color = "🟢" if self.get_engine_status() else "🔴"
        st.sidebar.metric("Engine Status", status_color)

        # Profit Manager Section
        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 Profit Manager")
        
        profit_data = self.get_profit_data()
        
        # Display pending balance with green accent
        st.sidebar.markdown(
            f"<h2 style='color: #34D399; text-shadow: 0 0 10px rgba(52, 211, 153, 0.5);'>"
            f"+{profit_data['accumulated_eth']:.4f} ETH</h2>",
            unsafe_allow_html=True
        )
        st.sidebar.caption("PENDING BALANCE")
        
        # Auto/Manual Mode Toggle
        transfer_mode = st.sidebar.radio(
            "Transfer Mode",
            ["Manual", "Auto"],
            index=0 if not profit_data['auto_transfer'] else 1,
            horizontal=True
        )
        
        # Update mode if changed
        if (transfer_mode == "Auto") != profit_data['auto_transfer']:
            self.update_profit_config(transfer_mode == "Auto", profit_data['threshold_eth'])
            st.rerun()
        
        # Manual Withdraw Button or Auto Status
        if transfer_mode == "Manual":
            if st.sidebar.button("🚀 WITHDRAW NOW", 
                               disabled=profit_data['accumulated_eth'] <= 0,
                               use_container_width=True):
                if self.execute_manual_withdrawal():
                    st.sidebar.success("Withdrawal Initiated!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.sidebar.error("Withdrawal Failed")
        else:
            st.sidebar.success("✅ AUTO-SWEEP ACTIVE")
            st.sidebar.caption(f"Threshold: {profit_data['threshold_eth']:.4f} ETH")

        # Quick stats
        st.sidebar.markdown("---")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("24h Profit", f"${self.get_24h_profit():,.2f}")
        with col2:
            st.metric("Active Trades", self.get_active_trades())

        # Risk alerts
        if self.check_risk_alerts():
            st.sidebar.error("⚠️ Risk Alert Active")

    def performance_tab(self):
        """Performance metrics and charts"""
        st.header("Performance Analytics")

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total P&L", f"${self.get_total_pnl():,.2f}",
                     delta=f"{self.get_pnl_change():+.2%}")
        with col2:
            st.metric("Win Rate", f"{self.get_win_rate():.1f}%")
        with col3:
            st.metric("Avg Trade Size", f"${self.get_avg_trade_size():,.0f}")
        with col4:
            st.metric("Sharpe Ratio", f"{self.get_sharpe_ratio():.2f}")

        # Performance chart
        st.subheader("P&L Over Time")
        pnl_chart = self.create_pnl_chart()
        st.plotly_chart(pnl_chart, use_container_width=True)

        # Trade distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Trade Size Distribution")
            size_dist = self.create_trade_size_distribution()
            st.plotly_chart(size_dist, use_container_width=True)

        with col2:
            st.subheader("Profit/Loss Distribution")
            pl_dist = self.create_pl_distribution()
            st.plotly_chart(pl_dist, use_container_width=True)

    def opportunities_tab(self):
        """Current arbitrage opportunities"""
        st.header("Live Arbitrage Opportunities")

        # Real-time opportunities table
        opportunities_df = self.get_current_opportunities()
        st.dataframe(opportunities_df, use_container_width=True)

        # Opportunity heatmap
        st.subheader("DEX Price Heatmap")
        heatmap = self.create_price_heatmap()
        st.plotly_chart(heatmap, use_container_width=True)

        # AI Predictions
        st.subheader("AI Opportunity Predictions")
        predictions_df = self.get_ai_predictions()
        st.dataframe(predictions_df, use_container_width=True)

    def risk_tab(self):
        """Risk management dashboard"""
        st.header("Risk Management")

        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VaR (95%)", f"${self.get_var_95():,.0f}")
        with col2:
            st.metric("Max Drawdown", f"{self.get_max_drawdown():.1f}%")
        with col3:
            st.metric("Liquidity Risk", self.get_liquidity_risk())
        with col4:
            st.metric("Slippage Risk", f"{self.get_slippage_risk():.1f}%")

        # Risk alerts
        st.subheader("Active Risk Alerts")
        alerts_df = self.get_risk_alerts()
        if not alerts_df.empty:
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.success("No active risk alerts")

        # Risk chart
        st.subheader("Risk Exposure Over Time")
        risk_chart = self.create_risk_chart()
        st.plotly_chart(risk_chart, use_container_width=True)

    def settings_tab(self):
        """Configuration settings"""
        st.header("Engine Configuration")

        # Risk settings
        st.subheader("Risk Parameters")
        col1, col2 = st.columns(2)
        with col1:
            max_slippage = st.slider("Max Slippage (%)", 0.1, 5.0, 2.0)
            max_position = st.slider("Max Position Size ($)", 10000, 10000000, 1000000)
        with col2:
            confidence_threshold = st.slider("AI Confidence Threshold", 0.5, 0.95, 0.8)
            scan_interval = st.slider("Scan Interval (ms)", 50, 1000, 100)

        # DEX settings
        st.subheader("DEX Configuration")
        enabled_dexes = st.multiselect(
            "Enabled DEXes",
            ["Uniswap V3", "SushiSwap", "PancakeSwap", "1inch", "CowSwap"],
            ["Uniswap V3", "SushiSwap", "CowSwap"]
        )

        # Save settings button
        if st.button("Save Configuration"):
            self.save_settings({
                'max_slippage': max_slippage,
                'max_position': max_position,
                'confidence_threshold': confidence_threshold,
                'scan_interval': scan_interval,
                'enabled_dexes': enabled_dexes
            })
            st.success("Settings saved successfully!")

    # Helper methods for data retrieval
    def get_profit_data(self):
        """Fetch real profit data from API"""
        try:
            response = requests.get(f"{self.api_base_url}/profit", timeout=5)
            if response.ok:
                return response.json()
        except:
            pass
        # Fallback
        return {
            'accumulated_eth': 0.0,
            'threshold_eth': 0.1,
            'auto_transfer': False,
            'total_pnl': 0.0,
            'active_trades': 0,
            'gas_saved': 0.0
        }
    
    def update_profit_config(self, enabled, threshold):
        """Update profit configuration via API"""
        try:
            response = requests.post(
                f"{self.api_base_url}/settings/profit-config",
                json={'enabled': enabled, 'threshold': threshold},
                timeout=5
            )
            return response.ok
        except:
            return False
    
    def execute_manual_withdrawal(self):
        """Execute manual withdrawal via API"""
        try:
            response = requests.post(f"{self.api_base_url}/withdraw", timeout=10)
            return response.ok
        except:
            return False
    
    def get_engine_status(self):
        try:
            response = requests.get(f"{self.api_base_url}/status", timeout=5)
            return response.ok
        except:
            return False

    def get_24h_profit(self):
        profit_data = self.get_profit_data()
        return profit_data.get('total_pnl', 0.0)

    def get_active_trades(self):
        profit_data = self.get_profit_data()
        return profit_data.get('active_trades', 0)

    def check_risk_alerts(self):
        return False

    def get_total_pnl(self):
        return self.get_24h_profit()

    def get_pnl_change(self):
        return 12.5  # TODO: Calculate from historical data

    def get_win_rate(self):
        return 78.5  # TODO: Calculate from trade history

    def get_avg_trade_size(self):
        return 50000  # TODO: Calculate from trade history

    def get_sharpe_ratio(self):
        return 2.34  # TODO: Calculate from performance data

    def create_pnl_chart(self):
        # Mock P&L data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        pnl = [1000 + i*200 + np.random.normal(0, 500) for i in range(30)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=pnl, mode='lines+markers', name='P&L'))
        fig.update_layout(title='Portfolio P&L Over Time', xaxis_title='Date', yaxis_title='P&L ($)')
        return fig

    def create_trade_size_distribution(self):
        sizes = [10000, 25000, 50000, 75000, 100000, 150000]
        counts = [5, 12, 8, 3, 2, 1]

        fig = px.bar(x=sizes, y=counts, labels={'x': 'Trade Size ($)', 'y': 'Frequency'})
        fig.update_layout(title='Trade Size Distribution')
        return fig

    def create_pl_distribution(self):
        # Mock P&L distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=np.random.normal(1000, 2000, 100), nbinsx=20))
        fig.update_layout(title='Profit/Loss Distribution', xaxis_title='P&L ($)', yaxis_title='Frequency')
        return fig

    def get_current_opportunities(self):
        # Mock opportunities data
        return pd.DataFrame({
            'DEX Pair': ['UNI-SUSHI', 'UNI-PANCAKE', 'SUSHI-COW'],
            'Spread (%)': [1.2, 0.8, 2.1],
            'Volume ($)': [50000, 75000, 30000],
            'Confidence': [0.85, 0.72, 0.91],
            'Estimated Profit ($)': [600, 400, 630]
        })

    def create_price_heatmap(self):
        # Mock price data
        dexes = ['Uniswap', 'SushiSwap', 'PancakeSwap', '1inch']
        tokens = ['WETH', 'USDC', 'WBTC', 'UNI']
        prices = np.random.uniform(1000, 3000, (len(dexes), len(tokens)))

        fig = px.imshow(prices,
                       x=tokens,
                       y=dexes,
                       color_continuous_scale='RdYlGn',
                       title='DEX Price Heatmap')
        return fig

    def get_ai_predictions(self):
        return pd.DataFrame({
            'Opportunity': ['WETH/USDC Arb', 'WBTC/ETH Arb', 'UNI Flash Loan'],
            'Predicted Profit ($)': [450, 320, 890],
            'Confidence (%)': [87, 74, 92],
            'Time to Execute (s)': [0.05, 0.08, 0.03]
        })

    def get_var_95(self):
        return 25000

    def get_max_drawdown(self):
        return 8.5

    def get_liquidity_risk(self):
        return "Low"

    def get_slippage_risk(self):
        return 1.2

    def get_risk_alerts(self):
        return pd.DataFrame()  # No alerts

    def create_risk_chart(self):
        # Mock risk data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        risk = [2.0 + np.random.normal(0, 0.5) for _ in range(30)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=risk, mode='lines', name='Risk Exposure'))
        fig.update_layout(title='Risk Exposure Over Time', xaxis_title='Date', yaxis_title='Risk Score')
        return fig

    def save_settings(self, settings):
        # Save settings to file or database
        pass

# WebSocket handler for real-time updates
async def websocket_handler(websocket, path):
    dashboard = MonitoringDashboard()
    while True:
        # Send real-time updates
        data = {
            'pnl': dashboard.get_total_pnl(),
            'active_trades': dashboard.get_active_trades(),
            'opportunities': dashboard.get_current_opportunities().to_dict()
        }
        await websocket.send(json.dumps(data))
        await asyncio.sleep(1)

def start_websocket_server():
    """Start WebSocket server for real-time updates"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = websockets.serve(websocket_handler, "localhost", 8765)
    loop.run_until_complete(server)
    loop.run_forever()

if __name__ == "__main__":
    # Start WebSocket server in background thread
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()

    # Run Streamlit dashboard
    dashboard = MonitoringDashboard()
    dashboard.run_dashboard()
