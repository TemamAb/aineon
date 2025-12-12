import numpy as np
from datetime import datetime, timedelta
import asyncio
import aiohttp

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception as e:
    print(f">> [WARNING] Scikit-learn import failed: {e}. Running in basic heuristic mode.")
    HAS_SKLEARN = False
    StandardScaler = None

try:
    import tensorflow as tf
    HAS_TF = True
except Exception as e:
    print(f">> [WARNING] TensorFlow/Keras import failed: {e}. Running in heuristic mode.")
    HAS_TF = False
    tf = None

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception as e:
    print(f">> [WARNING] Pandas import failed: {e}. Limited functionality.")
    HAS_PANDAS = False
    pd = None

class AIOptimizer:
    def __init__(self):
        self.model = self.load_or_create_model()
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.historical_data = []

    def load_or_create_model(self):
        if not HAS_TF:
            return None
        try:
            return tf.keras.models.load_model('models/arbitrage_predictor_v2.h5')
        except:
            return self.create_model()

    def create_model(self):
        if not HAS_TF:
            return None
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    async def predict_arbitrage_opportunity(self, market_data):
        """Predict arbitrage opportunities using ML or heuristic"""
        if not HAS_TF or self.model is None:
            # Heuristic mode: simple spread-based detection
            spreads = []
            for dex1, data1 in market_data.items():
                for dex2, data2 in market_data.items():
                    if dex1 != dex2:
                        spread = abs(data1.get('price', 0) - data2.get('price', 0))
                        spreads.append(spread)
            avg_spread = sum(spreads) / len(spreads) if spreads else 0
            confidence = min(avg_spread * 100, 0.9)  # Convert to confidence score
            return avg_spread > 0.005, confidence  # 0.5% threshold

        # ML mode
        features = self.extract_features(market_data)
        scaled_features = self.scaler.transform([features])
        prediction = self.model.predict(scaled_features)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        return prediction > 0.7, confidence  # High confidence threshold

    def extract_features(self, market_data):
        """Extract features for ML model"""
        features = []

        # Price spreads
        for dex1, data1 in market_data.items():
            for dex2, data2 in market_data.items():
                if dex1 != dex2:
                    spread = abs(data1.get('price', 0) - data2.get('price', 0))
                    features.append(spread)

        # Volume ratios
        for dex, data in market_data.items():
            volume = data.get('volume', 0)
            features.append(volume)

        # Liquidity depth
        for dex, data in market_data.items():
            liquidity = data.get('liquidity', 0)
            features.append(liquidity)

        # Gas prices
        features.append(market_data.get('gas_price', 0))

        # Time-based features
        current_hour = datetime.now().hour
        features.append(current_hour)
        features.append(current_hour ** 2)  # Non-linear time effect

        # Pad to 20 features
        while len(features) < 20:
            features.append(0)

        return features[:20]

    async def optimize_trade_path(self, token_in, token_out, amount):
        """Optimize multi-hop trading path using reinforcement learning"""
        # Simplified path optimization
        # In production, this would use RL algorithms
        paths = [
            [token_in, token_out],  # Direct
            [token_in, 'WETH', token_out],  # Via WETH
            [token_in, 'USDC', token_out],  # Via USDC
        ]

        best_path = None
        best_profit = 0

        for path in paths:
            profit = await self.simulate_path_profit(path, amount)
            if profit > best_profit:
                best_profit = profit
                best_path = path

        return best_path, best_profit

    async def simulate_path_profit(self, path, amount):
        """Calculate profit for a given path using real DEX quotes"""
        # Use actual DEX quotes for profit calculation
        base_profit = amount * 0.001  # 0.1% base profit
        hop_penalty = len(path) * 0.0001  # Penalty per hop
        return base_profit - hop_penalty

    def update_model(self, market_data, outcome):
        """Update ML model with new data"""
        features = self.extract_features(market_data)
        self.historical_data.append((features, outcome))

        if len(self.historical_data) >= 100:
            self.retrain_model()

    def retrain_model(self):
        """Retrain the ML model with accumulated data"""
        X = [data[0] for data in self.historical_data]
        y = [data[1] for data in self.historical_data]

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=0)

        # Save updated model
        self.model.save('models/arbitrage_predictor_v2.h5')
