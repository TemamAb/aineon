"""
AI-NEXUS v5.0 - MARKET PREDICTOR MODULE
Advanced Multi-Timeframe Market Prediction Engine
Combines statistical models, machine learning, and on-chain analytics for price forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

class PredictionHorizon(Enum):
    SHORT_TERM = "short_term"  # 1-15 minutes
    MEDIUM_TERM = "medium_term"  # 1-4 hours
    LONG_TERM = "long_term"  # 1-7 days

class PredictionModel(Enum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"
    STATISTICAL = "statistical"

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    VOLATILE = "volatile"
    RANGING = "ranging"
    BREAKOUT = "breakout"

@dataclass
class PricePrediction:
    prediction_id: str
    timestamp: datetime
    asset: str
    horizon: PredictionHorizon
    predicted_price: float
    confidence: float
    prediction_interval: Tuple[float, float]  # Lower and upper bounds
    model_used: PredictionModel
    features_used: List[str]
    metadata: Dict[str, Any]

@dataclass
class MarketRegimePrediction:
    regime_id: str
    timestamp: datetime
    current_regime: MarketRegime
    next_regime: MarketRegime
    transition_probability: float
    expected_duration: timedelta
    confidence: float
    triggers: List[str]

class MarketPredictor:
    """
    Advanced market prediction system combining multiple models and data sources
    """
    
    def __init__(self):
        self.prediction_history = deque(maxlen=10000)
        self.model_registry = {}
        self.feature_engine = FeatureEngine()
        
        # Prediction parameters
        self.prediction_params = {
            'short_term_horizon': timedelta(minutes=15),
            'medium_term_horizon': timedelta(hours=4),
            'long_term_horizon': timedelta(days=7),
            'min_confidence_threshold': 0.6,
            'max_prediction_interval': 0.1,  # 10% price range
            'ensemble_weight_update_frequency': 1000
        }
        
        # Model configurations
        self.model_configs = {
            PredictionModel.LSTM: {
                'window_size': 50,
                'hidden_units': 128,
                'dropout_rate': 0.2,
                'training_interval': timedelta(hours=1)
            },
            PredictionModel.TRANSFORMER: {
                'num_layers': 4,
                'd_model': 64,
                'num_heads': 8,
                'window_size': 100
            },
            PredictionModel.GRADIENT_BOOSTING: {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'feature_subsample': 0.8
            },
            PredictionModel.ENSEMBLE: {
                'model_weights': {
                    PredictionModel.LSTM: 0.4,
                    PredictionModel.TRANSFORMER: 0.3,
                    PredictionModel.GRADIENT_BOOSTING: 0.3
                },
                'dynamic_weighting': True
            }
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'accurate_predictions': 0,
            'avg_confidence': 0.0,
            'model_performance': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'feature_importance': {}
        }
        
        # Initialize models
        self._initialize_models()
        self._initialize_regime_detector()
    
    def _initialize_models(self):
        """Initialize prediction models"""
        
        # Placeholder for actual model initialization
        # In production, these would be properly trained models
        
        self.model_registry = {
            PredictionModel.LSTM: {
                'model': None,  # Would be actual LSTM model
                'last_trained': datetime.now(),
                'performance': 0.0,
                'is_trained': False
            },
            PredictionModel.TRANSFORMER: {
                'model': None,  # Would be actual transformer model
                'last_trained': datetime.now(),
                'performance': 0.0,
                'is_trained': False
            },
            PredictionModel.GRADIENT_BOOSTING: {
                'model': None,  # Would be actual XGBoost/LightGBM model
                'last_trained': datetime.now(),
                'performance': 0.0,
                'is_trained': False
            },
            PredictionModel.ENSEMBLE: {
                'model': None,  # Ensemble combination
                'last_trained': datetime.now(),
                'performance': 0.0,
                'is_trained': False
            },
            PredictionModel.STATISTICAL: {
                'model': None,  # Statistical models (ARIMA, GARCH)
                'last_trained': datetime.now(),
                'performance': 0.0,
                'is_trained': True  # Statistical models don't need traditional training
            }
        }
        
        print("Market prediction models initialized")
    
    def _initialize_regime_detector(self):
        """Initialize market regime detection system"""
        
        self.regime_detector = {
            'current_regime': MarketRegime.RANGING,
            'regime_history': deque(maxlen=1000),
            'transition_matrix': self._initialize_transition_matrix(),
            'regime_features': [
                'price_trend_strength',
                'volatility_level',
                'volume_trend',
                'market_momentum',
                'mean_reversion_signal'
            ]
        }
    
    def _initialize_transition_matrix(self) -> Dict[MarketRegime, Dict[MarketRegime, float]]:
        """Initialize market regime transition probabilities"""
        
        return {
            MarketRegime.TRENDING_UP: {
                MarketRegime.TRENDING_UP: 0.7,
                MarketRegime.TRENDING_DOWN: 0.1,
                MarketRegime.VOLATILE: 0.1,
                MarketRegime.RANGING: 0.1,
                MarketRegime.BREAKOUT: 0.0
            },
            MarketRegime.TRENDING_DOWN: {
                MarketRegime.TRENDING_UP: 0.1,
                MarketRegime.TRENDING_DOWN: 0.7,
                MarketRegime.VOLATILE: 0.1,
                MarketRegime.RANGING: 0.1,
                MarketRegime.BREAKOUT: 0.0
            },
            MarketRegime.VOLATILE: {
                MarketRegime.TRENDING_UP: 0.2,
                MarketRegime.TRENDING_DOWN: 0.2,
                MarketRegime.VOLATILE: 0.4,
                MarketRegime.RANGING: 0.2,
                MarketRegime.BREAKOUT: 0.0
            },
            MarketRegime.RANGING: {
                MarketRegime.TRENDING_UP: 0.2,
                MarketRegime.TRENDING_DOWN: 0.2,
                MarketRegime.VOLATILE: 0.2,
                MarketRegime.RANGING: 0.3,
                MarketRegime.BREAKOUT: 0.1
            },
            MarketRegime.BREAKOUT: {
                MarketRegime.TRENDING_UP: 0.4,
                MarketRegime.TRENDING_DOWN: 0.4,
                MarketRegime.VOLATILE: 0.1,
                MarketRegime.RANGING: 0.1,
                MarketRegime.BREAKOUT: 0.0
            }
        }
    
    async def predict_price(self, 
                          asset: str,
                          horizon: PredictionHorizon,
                          market_data: Dict[str, Any],
                          model_preference: PredictionModel = None) -> PricePrediction:
        """Generate price prediction for specified asset and horizon"""
        
        # Extract and engineer features
        features = await self.feature_engine.engineer_features(asset, market_data)
        
        # Select appropriate model
        model_to_use = self._select_prediction_model(horizon, model_preference)
        
        # Generate prediction
        if model_to_use == PredictionModel.ENSEMBLE:
            prediction = await self._ensemble_prediction(asset, horizon, features)
        else:
            prediction = await self._single_model_prediction(asset, horizon, features, model_to_use)
        
        # Update prediction history
        self.prediction_history.append(prediction)
        self.performance_metrics['total_predictions'] += 1
        
        return prediction
    
    async def _ensemble_prediction(self, 
                                 asset: str,
                                 horizon: PredictionHorizon,
                                 features: Dict[str, np.ndarray]) -> PricePrediction:
        """Generate ensemble prediction combining multiple models"""
        
        predictions = []
        confidences = []
        model_predictions = {}
        
        # Get predictions from all models
        for model_type in [PredictionModel.LSTM, PredictionModel.TRANSFORMER, 
                          PredictionModel.GRADIENT_BOOSTING, PredictionModel.STATISTICAL]:
            
            if self.model_registry[model_type]['is_trained']:
                try:
                    model_pred = await self._single_model_prediction(
                        asset, horizon, features, model_type, store_history=False
                    )
                    predictions.append(model_pred.predicted_price)
                    confidences.append(model_pred.confidence)
                    model_predictions[model_type.value] = model_pred.predicted_price
                except Exception as e:
                    print(f"Model {model_type.value} prediction failed: {e}")
                    continue
        
        if not predictions:
            raise Exception("No models available for ensemble prediction")
        
        # Weighted average based on model performance and confidence
        weights = self._calculate_ensemble_weights(confidences, [PredictionModel(m) for m in model_predictions.keys()])
        weighted_prediction = np.average(predictions, weights=weights)
        
        # Calculate ensemble confidence
        ensemble_confidence = np.average(confidences, weights=weights)
        
        # Calculate prediction interval
        prediction_interval = self._calculate_prediction_interval(predictions, weighted_prediction, horizon)
        
        prediction = PricePrediction(
            prediction_id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(),
            asset=asset,
            horizon=horizon,
            predicted_price=weighted_prediction,
            confidence=ensemble_confidence,
            prediction_interval=prediction_interval,
            model_used=PredictionModel.ENSEMBLE,
            features_used=list(features.keys()),
            metadata={
                'component_predictions': model_predictions,
                'model_weights': dict(zip([m.value for m in model_predictions.keys()], weights)),
                'ensemble_method': 'weighted_average'
            }
        )
        
        return prediction
    
    async def _single_model_prediction(self,
                                     asset: str,
                                     horizon: PredictionHorizon,
                                     features: Dict[str, np.ndarray],
                                     model_type: PredictionModel,
                                     store_history: bool = True) -> PricePrediction:
        """Generate prediction using a single model"""
        
        model_info = self.model_registry[model_type]
        
        if not model_info['is_trained']:
            raise Exception(f"Model {model_type.value} is not trained")
        
        # Prepare features for specific model
        model_features = self._prepare_model_features(features, model_type, horizon)
        
        # Generate prediction (placeholder implementation)
        # In production, this would use actual model inference
        current_price = features.get('current_price', 1000)
        
        # Simulate model prediction based on features
        predicted_price = self._simulate_model_prediction(current_price, features, model_type, horizon)
        
        # Calculate confidence
        confidence = self._calculate_prediction_confidence(features, model_type, horizon)
        
        # Calculate prediction interval
        prediction_interval = self._calculate_prediction_interval([predicted_price], predicted_price, horizon)
        
        prediction = PricePrediction(
            prediction_id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(),
            asset=asset,
            horizon=horizon,
            predicted_price=predicted_price,
            confidence=confidence,
            prediction_interval=prediction_interval,
            model_used=model_type,
            features_used=list(features.keys()),
            metadata={
                'model_version': 'v1.0',
                'feature_count': len(model_features),
                'training_recency': model_info['last_trained'].isoformat()
            }
        )
        
        if store_history:
            self.prediction_history.append(prediction)
        
        return prediction
    
    def _simulate_model_prediction(self, 
                                 current_price: float,
                                 features: Dict[str, np.ndarray],
                                 model_type: PredictionModel,
                                 horizon: PredictionHorizon) -> float:
        """Simulate model prediction (placeholder for actual model inference)"""
        
        # Extract relevant features for prediction
        price_trend = features.get('price_momentum_15m', 0)
        volatility = features.get('volatility_1h', 0.1)
        volume_trend = features.get('volume_momentum', 0)
        market_sentiment = features.get('market_sentiment', 0.5)
        
        # Model-specific prediction logic
        if model_type == PredictionModel.LSTM:
            # LSTM would capture complex temporal patterns
            prediction = current_price * (1 + price_trend * 0.8 + volume_trend * 0.2)
        
        elif model_type == PredictionModel.TRANSFORMER:
            # Transformer would capture long-range dependencies
            prediction = current_price * (1 + price_trend * 0.7 + market_sentiment * 0.3)
        
        elif model_type == PredictionModel.GRADIENT_BOOSTING:
            # Gradient boosting would use feature interactions
            prediction = current_price * (1 + price_trend * 0.6 + volume_trend * 0.2 + market_sentiment * 0.2)
        
        elif model_type == PredictionModel.STATISTICAL:
            # Statistical model based on mean reversion and momentum
            mean_reversion = features.get('mean_reversion_signal', 0)
            prediction = current_price * (1 + price_trend * 0.5 + mean_reversion * 0.3)
        
        else:
            prediction = current_price * (1 + price_trend * 0.5)
        
        # Add noise based on volatility and horizon
        horizon_multiplier = self._get_horizon_multiplier(horizon)
        noise = np.random.normal(0, volatility * horizon_multiplier)
        prediction = prediction * (1 + noise * 0.1)
        
        return float(prediction)
    
    def _calculate_prediction_confidence(self,
                                      features: Dict[str, np.ndarray],
                                      model_type: PredictionModel,
                                      horizon: PredictionHorizon) -> float:
        """Calculate prediction confidence based on market conditions and model performance"""
        
        base_confidence = 0.7
        
        # Adjust based on volatility
        volatility = features.get('volatility_1h', 0.1)
        volatility_penalty = min(0.3, volatility * 2)
        
        # Adjust based on feature quality
        feature_quality = self._assess_feature_quality(features)
        
        # Adjust based on model performance
        model_performance = self.model_registry[model_type]['performance']
        model_bonus = model_performance * 0.2
        
        # Adjust based on horizon (shorter horizons generally more predictable)
        horizon_factor = self._get_horizon_confidence_factor(horizon)
        
        confidence = (base_confidence - volatility_penalty + feature_quality + model_bonus) * horizon_factor
        
        return max(0.1, min(0.95, confidence))
    
    def _calculate_prediction_interval(self,
                                    predictions: List[float],
                                    central_prediction: float,
                                    horizon: PredictionHorizon) -> Tuple[float, float]:
        """Calculate prediction interval based on model variance and horizon"""
        
        if len(predictions) == 0:
            return (central_prediction * 0.95, central_prediction * 1.05)
        
        # Calculate interval based on prediction variance
        prediction_std = np.std(predictions) if len(predictions) > 1 else central_prediction * 0.02
        
        # Adjust interval based on horizon
        horizon_multiplier = self._get_horizon_interval_multiplier(horizon)
        interval_width = prediction_std * horizon_multiplier
        
        lower_bound = central_prediction * (1 - interval_width)
        upper_bound = central_prediction * (1 + interval_width)
        
        # Ensure bounds are reasonable
        max_interval = self.prediction_params['max_prediction_interval']
        lower_bound = max(central_prediction * (1 - max_interval), lower_bound)
        upper_bound = min(central_prediction * (1 + max_interval), upper_bound)
        
        return (float(lower_bound), float(upper_bound))
    
    def _calculate_ensemble_weights(self, 
                                  confidences: List[float],
                                  model_types: List[PredictionModel]) -> List[float]:
        """Calculate weights for ensemble prediction"""
        
        weights = []
        
        for confidence, model_type in zip(confidences, model_types):
            # Base weight from model performance
            model_perf = self.model_registry[model_type]['performance']
            
            # Combine confidence and performance
            weight = confidence * 0.7 + model_perf * 0.3
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights
    
    def _select_prediction_model(self, 
                               horizon: PredictionHorizon,
                               model_preference: PredictionModel = None) -> PredictionModel:
        """Select appropriate prediction model based on horizon and preferences"""
        
        if model_preference:
            return model_preference
        
        # Model selection based on horizon
        if horizon == PredictionHorizon.SHORT_TERM:
            return PredictionModel.LSTM  # Good for short-term patterns
        elif horizon == PredictionHorizon.MEDIUM_TERM:
            return PredictionModel.ENSEMBLE  # Balanced approach
        else:  # LONG_TERM
            return PredictionModel.GRADIENT_BOOSTING  # Good for structural patterns
    
    def _prepare_model_features(self, 
                              features: Dict[str, np.ndarray],
                              model_type: PredictionModel,
                              horizon: PredictionHorizon) -> np.ndarray:
        """Prepare features for specific model type"""
        
        # Select relevant features based on model type and horizon
        feature_subset = []
        
        # Common features for all models
        common_features = [
            'price_momentum_5m', 'price_momentum_15m', 'price_momentum_1h',
            'volatility_15m', 'volatility_1h', 'volatility_4h',
            'volume_momentum', 'volume_volatility',
            'market_sentiment', 'mean_reversion_signal'
        ]
        
        feature_subset.extend(common_features)
        
        # Model-specific features
        if model_type == PredictionModel.LSTM:
            feature_subset.extend([
                'price_sequence_50', 'volume_sequence_50',
                'technical_indicators_sequence'
            ])
        elif model_type == PredictionModel.TRANSFORMER:
            feature_subset.extend([
                'price_sequence_100', 'volume_sequence_100',
                'attention_features'
            ])
        elif model_type == PredictionModel.GRADIENT_BOOSTING:
            feature_subset.extend([
                'price_features_engineered', 'volume_features_engineered',
                'market_structure_features'
            ])
        
        # Extract selected features
        model_features = []
        for feature_name in feature_subset:
            if feature_name in features:
                feature_data = features[feature_name]
                if isinstance(feature_data, (int, float)):
                    model_features.append(feature_data)
                elif isinstance(feature_data, np.ndarray):
                    model_features.extend(feature_data.flatten())
        
        return np.array(model_features)
    
    def _assess_feature_quality(self, features: Dict[str, np.ndarray]) -> float:
        """Assess quality of features for prediction"""
        
        quality_score = 0.5  # Base score
        
        # Check feature completeness
        expected_features = ['current_price', 'price_momentum_15m', 'volatility_1h', 'volume_momentum']
        available_features = [f for f in expected_features if f in features]
        completeness = len(available_features) / len(expected_features)
        
        # Check feature recency
        data_freshness = features.get('data_freshness', 1.0)
        
        # Check feature stability
        feature_variance = features.get('feature_stability', 0.8)
        
        quality_score = (completeness * 0.4 + data_freshness * 0.3 + feature_variance * 0.3)
        
        return quality_score
    
    def _get_horizon_multiplier(self, horizon: PredictionHorizon) -> float:
        """Get multiplier for horizon-based adjustments"""
        if horizon == PredictionHorizon.SHORT_TERM:
            return 1.0
        elif horizon == PredictionHorizon.MEDIUM_TERM:
            return 2.0
        else:  # LONG_TERM
            return 4.0
    
    def _get_horizon_confidence_factor(self, horizon: PredictionHorizon) -> float:
        """Get confidence factor based on prediction horizon"""
        if horizon == PredictionHorizon.SHORT_TERM:
            return 1.0
        elif horizon == PredictionHorizon.MEDIUM_TERM:
            return 0.8
        else:  # LONG_TERM
            return 0.6
    
    def _get_horizon_interval_multiplier(self, horizon: PredictionHorizon) -> float:
        """Get interval multiplier based on prediction horizon"""
        if horizon == PredictionHorizon.SHORT_TERM:
            return 1.0
        elif horizon == PredictionHorizon.MEDIUM_TERM:
            return 1.5
        else:  # LONG_TERM
            return 2.0
    
    async def predict_market_regime(self, market_data: Dict[str, Any]) -> MarketRegimePrediction:
        """Predict current and future market regimes"""
        
        # Extract regime features
        regime_features = self._extract_regime_features(market_data)
        
        # Detect current regime
        current_regime = self._detect_current_regime(regime_features)
        
        # Predict next regime
        next_regime, transition_prob = self._predict_next_regime(current_regime, regime_features)
        
        # Calculate expected duration
        expected_duration = self._calculate_expected_regime_duration(current_regime)
        
        # Calculate confidence
        confidence = self._calculate_regime_confidence(regime_features, current_regime)
        
        # Identify regime triggers
        triggers = self._identify_regime_triggers(regime_features, current_regime)
        
        regime_prediction = MarketRegimePrediction(
            regime_id=f"regime_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=datetime.now(),
            current_regime=current_regime,
            next_regime=next_regime,
            transition_probability=transition_prob,
            expected_duration=expected_duration,
            confidence=confidence,
            triggers=triggers
        )
        
        # Update regime history
        self.regime_detector['current_regime'] = current_regime
        self.regime_detector['regime_history'].append(regime_prediction)
        
        return regime_prediction
    
    def _extract_regime_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for regime detection"""
        
        features = {}
        
        # Price trend strength
        features['price_trend_strength'] = market_data.get('price_trend_strength', 0)
        
        # Volatility level
        features['volatility_level'] = market_data.get('volatility_1h', 0.1)
        
        # Volume trend
        features['volume_trend'] = market_data.get('volume_momentum', 0)
        
        # Market momentum
        features['market_momentum'] = market_data.get('market_momentum', 0)
        
        # Mean reversion signal
        features['mean_reversion_signal'] = market_data.get('mean_reversion_signal', 0)
        
        return features
    
    def _detect_current_regime(self, regime_features: Dict[str, float]) -> MarketRegime:
        """Detect current market regime based on features"""
        
        trend_strength = regime_features['price_trend_strength']
        volatility = regime_features['volatility_level']
        volume_trend = regime_features['volume_trend']
        
        if volatility > 0.15:
            return MarketRegime.VOLATILE
        elif abs(trend_strength) > 0.7:
            return MarketRegime.TRENDING_UP if trend_strength > 0 else MarketRegime.TRENDING_DOWN
        elif volume_trend > 0.8:
            return MarketRegime.BREAKOUT
        else:
            return MarketRegime.RANGING
    
    def _predict_next_regime(self, 
                           current_regime: MarketRegime,
                           regime_features: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Predict next market regime and transition probability"""
        
        # Get base transition probabilities
        transition_probs = self.regime_detector['transition_matrix'][current_regime]
        
        # Adjust probabilities based on current features
        volatility = regime_features['volatility_level']
        if volatility > 0.2:
            # Increase probability of volatile regime
            transition_probs[MarketRegime.VOLATILE] *= 1.5
        
        # Normalize probabilities
        total_prob = sum(transition_probs.values())
        normalized_probs = {k: v/total_prob for k, v in transition_probs.items()}
        
        # Select most likely next regime
        next_regime = max(normalized_probs.items(), key=lambda x: x[1])[0]
        transition_prob = normalized_probs[next_regime]
        
        return next_regime, transition_prob
    
    def _calculate_expected_regime_duration(self, current_regime: MarketRegime) -> timedelta:
        """Calculate expected duration of current regime"""
        
        # Base durations for different regimes
        base_durations = {
            MarketRegime.TRENDING_UP: timedelta(hours=6),
            MarketRegime.TRENDING_DOWN: timedelta(hours=6),
            MarketRegime.VOLATILE: timedelta(hours=2),
            MarketRegime.RANGING: timedelta(hours=8),
            MarketRegime.BREAKOUT: timedelta(hours=1)
        }
        
        return base_durations.get(current_regime, timedelta(hours=4))
    
    def _calculate_regime_confidence(self, 
                                  regime_features: Dict[str, float],
                                  detected_regime: MarketRegime) -> float:
        """Calculate confidence in regime detection"""
        
        confidence = 0.7  # Base confidence
        
        # Adjust based on feature clarity
        trend_strength = abs(regime_features['price_trend_strength'])
        volatility = regime_features['volatility_level']
        
        if detected_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            confidence *= min(1.0, trend_strength * 1.5)
        elif detected_regime == MarketRegime.VOLATILE:
            confidence *= min(1.0, volatility * 5)
        
        return max(0.3, min(0.95, confidence))
    
    def _identify_regime_triggers(self, 
                                regime_features: Dict[str, float],
                                current_regime: MarketRegime) -> List[str]:
        """Identify triggers that caused current regime"""
        
        triggers = []
        
        trend_strength = regime_features['price_trend_strength']
        volatility = regime_features['volatility_level']
        volume_trend = regime_features['volume_trend']
        
        if current_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            if abs(trend_strength) > 0.8:
                triggers.append("Strong price momentum")
            if volume_trend > 0.7:
                triggers.append("High volume confirmation")
        
        elif current_regime == MarketRegime.VOLATILE:
            if volatility > 0.2:
                triggers.append("Elevated market volatility")
            if abs(trend_strength) < 0.3:
                triggers.append("Lack of clear trend")
        
        elif current_regime == MarketRegime.RANGING:
            if abs(trend_strength) < 0.4:
                triggers.append("Weak price momentum")
            if volatility < 0.1:
                triggers.append("Low volatility conditions")
        
        elif current_regime == MarketRegime.BREAKOUT:
            if volume_trend > 0.8:
                triggers.append("Volume surge")
            if abs(trend_strength) > 0.6:
                triggers.append("Price breakout")
        
        return triggers
    
    def update_prediction_accuracy(self, prediction_id: str, actual_price: float):
        """Update model performance based on prediction accuracy"""
        
        prediction = next((p for p in self.prediction_history if p.prediction_id == prediction_id), None)
        
        if prediction:
            # Calculate prediction error
            error = abs(prediction.predicted_price - actual_price) / actual_price
            is_accurate = error < 0.02  # 2% error threshold
            
            # Update performance metrics
            if is_accurate:
                self.performance_metrics['accurate_predictions'] += 1
            
            # Update model-specific performance
            model_type = prediction.model_used
            self.performance_metrics['model_performance'][model_type.value]['total'] += 1
            if is_accurate:
                self.performance_metrics['model_performance'][model_type.value]['correct'] += 1
            
            # Update model registry performance
            model_perf = self.performance_metrics['model_performance'][model_type.value]
            accuracy = model_perf['correct'] / model_perf['total'] if model_perf['total'] > 0 else 0.0
            self.model_registry[model_type]['performance'] = accuracy
            
            # Update average confidence
            total_predictions = self.performance_metrics['total_predictions']
            current_avg = self.performance_metrics['avg_confidence']
            new_avg = (current_avg * (total_predictions - 1) + prediction.confidence) / total_predictions
            self.performance_metrics['avg_confidence'] = new_avg
            
            print(f"Prediction {prediction_id} accuracy updated: {'Accurate' if is_accurate else 'Inaccurate'}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get prediction performance metrics"""
        
        overall_accuracy = (
            self.performance_metrics['accurate_predictions'] / 
            self.performance_metrics['total_predictions']
            if self.performance_metrics['total_predictions'] > 0 else 0.0
        )
        
        model_accuracies = {}
        for model_type, perf in self.performance_metrics['model_performance'].items():
            if perf['total'] > 0:
                model_accuracies[model_type] = perf['correct'] / perf['total']
        
        return {
            'total_predictions': self.performance_metrics['total_predictions'],
            'overall_accuracy': overall_accuracy,
            'average_confidence': self.performance_metrics['avg_confidence'],
            'model_accuracies': model_accuracies,
            'current_regime': self.regime_detector['current_regime'].value
        }

class FeatureEngine:
    """Feature engineering for market prediction"""
    
    async def engineer_features(self, asset: str, market_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Engineer features for market prediction"""
        
        features = {}
        
        # Price-based features
        features.update(self._engineer_price_features(market_data.get('price_data', {}).get(asset, {})))
        
        # Volume-based features
        features.update(self._engineer_volume_features(market_data.get('volume_data', {}).get(asset, {})))
        
        # Market structure features
        features.update(self._engineer_market_structure_features(market_data))
        
        # Technical indicators
        features.update(self._calculate_technical_indicators(market_data.get('price_data', {}).get(asset, {})))
        
        # On-chain features (if available)
        features.update(self._engineer_onchain_features(market_data.get('onchain_data', {})))
        
        # Sentiment features
        features.update(self._engineer_sentiment_features(market_data.get('sentiment_data', {})))
        
        return features
    
    def _engineer_price_features(self, price_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Engineer price-based features"""
        
        features = {}
        
        current_price = price_data.get('current_price', 1000)
        price_history = price_data.get('price_history', [])
        
        if len(price_history) >= 10:
            prices = np.array(price_history)
            
            # Momentum features
            features['price_momentum_5m'] = self._calculate_momentum(prices, 5)
            features['price_momentum_15m'] = self._calculate_momentum(prices, 15)
            features['price_momentum_1h'] = self._calculate_momentum(prices, 60)
            
            # Volatility features
            features['volatility_15m'] = self._calculate_volatility(prices, 15)
            features['volatility_1h'] = self._calculate_volatility(prices, 60)
            features['volatility_4h'] = self._calculate_volatility(prices, 240)
            
            # Mean reversion signal
            features['mean_reversion_signal'] = self._calculate_mean_reversion_signal(prices)
            
            # Price sequences for sequence models
            features['price_sequence_50'] = prices[-50:] if len(prices) >= 50 else prices
            features['price_sequence_100'] = prices[-100:] if len(prices) >= 100 else prices
        
        features['current_price'] = current_price
        
        return features
    
    def _engineer_volume_features(self, volume_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Engineer volume-based features"""
        
        features = {}
        
        current_volume = volume_data.get('current_volume', 0)
        volume_history = volume_data.get('volume_history', [])
        
        if len(volume_history) >= 10:
            volumes = np.array(volume_history)
            
            # Volume momentum
            features['volume_momentum'] = self._calculate_momentum(volumes, 10)
            
            # Volume volatility
            features['volume_volatility'] = self._calculate_volatility(volumes, 10)
            
            # Volume sequences
            features['volume_sequence_50'] = volumes[-50:] if len(volumes) >= 50 else volumes
        
        features['current_volume'] = current_volume
        
        return features
    
    def _engineer_market_structure_features(self, market_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Engineer market structure features"""
        
        features = {}
        
        # Liquidity features
        liquidity_data = market_data.get('liquidity_data', {})
        if liquidity_data:
            features['liquidity_depth'] = np.mean([d.get('current_liquidity', 0) for d in liquidity_data.values()])
            features['liquidity_stability'] = self._calculate_liquidity_stability(liquidity_data)
        
        # Spread features
        features['bid_ask_spread'] = market_data.get('spread', 0.001)
        
        # Market depth
        features['market_depth'] = market_data.get('market_depth', 0.8)
        
        return features
    
    def _calculate_technical_indicators(self, price_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Calculate technical indicators"""
        
        features = {}
        
        price_history = price_data.get('price_history', [])
        if len(price_history) < 20:
            return features
        
        prices = np.array(price_history)
        
        # Simple Moving Averages
        features['sma_20'] = np.mean(prices[-20:])
        features['sma_50'] = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
        
        # RSI
        features['rsi_14'] = self._calculate_rsi(prices, 14)
        
        # MACD
        macd, signal = self._calculate_macd(prices)
        features['macd'] = macd
        features['macd_signal'] = signal
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(prices, 20)
        features['bollinger_position'] = (prices[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        return features
    
    def _engineer_onchain_features(self, onchain_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Engineer on-chain features"""
        
        features = {}
        
        # Transaction volume
        features['onchain_tx_volume'] = onchain_data.get('transaction_volume', 0)
        
        # Active addresses
        features['active_addresses'] = onchain_data.get('active_addresses', 0)
        
        # Exchange flows
        features['exchange_inflow'] = onchain_data.get('exchange_inflow', 0)
        features['exchange_outflow'] = onchain_data.get('exchange_outflow', 0)
        
        return features
    
    def _engineer_sentiment_features(self, sentiment_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Engineer sentiment features"""
        
        features = {}
        
        features['market_sentiment'] = sentiment_data.get('overall_sentiment', 0.5)
        features['social_volume'] = sentiment_data.get('social_volume', 0)
        features['fear_greed_index'] = sentiment_data.get('fear_greed_index', 50)
        
        return features
    
    def _calculate_momentum(self, data: np.ndarray, period: int) -> float:
        """Calculate momentum over specified period"""
        
        if len(data) < period:
            return 0.0
        
        return (data[-1] - data[-period]) / data[-period]
    
    def _calculate_volatility(self, data: np.ndarray, period: int) -> float:
        """Calculate volatility over specified period"""
        
        if len(data) < period:
            return 0.0
        
        returns = np.diff(data[-period:]) / data[-period:-1]
        return np.std(returns) if len(returns) > 0 else 0.0
    
    def _calculate_mean_reversion_signal(self, prices: np.ndarray) -> float:
        """Calculate mean reversion signal"""
        
        if len(prices) < 20:
            return 0.0
        
        current_price = prices[-1]
        mean_price = np.mean(prices[-20:])
        std_price = np.std(prices[-20:])
        
        if std_price == 0:
            return 0.0
        
        z_score = (current_price - mean_price) / std_price
        # Negative z-score suggests oversold (buy signal), positive suggests overbought (sell signal)
        return -z_score * 0.1  # Scale to reasonable range
    
    def _calculate_liquidity_stability(self, liquidity_data: Dict[str, Any]) -> float:
        """Calculate liquidity stability score"""
        
        stability_scores = []
        
        for pool_data in liquidity_data.values():
            liquidity_history = pool_data.get('liquidity_history', [])
            if len(liquidity_history) >= 5:
                volatility = np.std(liquidity_history) / np.mean(liquidity_history) if np.mean(liquidity_history) > 0 else 0
                stability = 1.0 / (1.0 + volatility * 10)  # Convert to stability score
                stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 0.8
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate Relative Strength Index"""
        
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float]:
        """Calculate MACD and signal line"""
        
        if len(prices) < 26:
            return 0.0, 0.0
        
        # Simplified MACD calculation
        ema_12 = np.mean(prices[-12:])  # Simplified EMA
        ema_26 = np.mean(prices[-26:])  # Simplified EMA
        
        macd = ema_12 - ema_26
        signal = np.mean(prices[-9:])  # Simplified signal line
        
        return float(macd), float(signal)
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        
        if len(prices) < period:
            return prices[-1] * 1.1, prices[-1] * 0.9
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        return float(upper_band), float(lower_band)

# Example usage
if __name__ == "__main__":
    # Create market predictor
    predictor = MarketPredictor()
    
    # Sample market data
    sample_market_data = {
        'price_data': {
            'ETH': {
                'current_price': 2500.0,
                'price_history': list(range(2400, 2510))  # Sample price history
            }
        },
        'volume_data': {
            'ETH': {
                'current_volume': 1000000,
                'volume_history': [800000, 850000, 900000, 950000, 1000000] * 20
            }
        },
        'liquidity_data': {
            'pool_1': {
                'current_liquidity': 5000000,
                'liquidity_history': [4800000, 4900000, 5000000, 5100000, 5000000]
            }
        },
        'onchain_data': {
            'transaction_volume': 1500000,
            'active_addresses': 50000,
            'exchange_inflow': 300000,
            'exchange_outflow': 250000
        },
        'sentiment_data': {
            'overall_sentiment': 0.7,
            'social_volume': 0.8,
            'fear_greed_index': 65
        }
    }
    
    # Generate predictions
    async def demo():
        # Price prediction
        price_pred = await predictor.predict_price(
            asset='ETH',
            horizon=PredictionHorizon.SHORT_TERM,
            market_data=sample_market_data
        )
        
        print(f"Price Prediction: ${price_pred.predicted_price:.2f}")
        print(f"Confidence: {price_pred.confidence:.2f}")
        print(f"Interval: ${price_pred.prediction_interval[0]:.2f} - ${price_pred.prediction_interval[1]:.2f}")
        
        # Regime prediction
        regime_pred = await predictor.predict_market_regime(sample_market_data)
        
        print(f"\nMarket Regime: {regime_pred.current_regime.value} -> {regime_pred.next_regime.value}")
        print(f"Transition Probability: {regime_pred.transition_probability:.2f}")
        print(f"Confidence: {regime_pred.confidence:.2f}")
        
        # Performance metrics
        metrics = predictor.get_performance_metrics()
        print(f"\nPerformance Metrics: {metrics}")
    
    import asyncio
    asyncio.run(demo())
