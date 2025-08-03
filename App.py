#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import redis
import pymongo
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import logging
from web3 import Web3
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/defi-lending')
WEB3_PROVIDER_URL = os.getenv('WEB3_PROVIDER_URL', 'http://localhost:8545')

# Initialize connections
redis_client = redis.from_url(REDIS_URL)
mongo_client = pymongo.MongoClient(MONGODB_URL)
db = mongo_client.get_default_database()
w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URL))

# ML Models
liquidation_model = None
price_prediction_model = None
risk_model = None
yield_optimization_model = None

# Model scalers
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

class LiquidationPredictor:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, positions_data):
        """Extract features for liquidation prediction"""
        features = []
        
        for position in positions_data:
            feature_vector = [
                float(position.get('health_factor', 1.0)),
                float(position.get('collateral_value_usd', 0)),
                float(position.get('debt_value_usd', 0)),
                float(position.get('utilization_rate', 0)),
                len(position.get('assets', [])),  # Number of assets
                float(position.get('largest_position_ratio', 0)),
                float(position.get('volatility_score', 0)),
                float(position.get('liquidity_score', 0)),
                float(position.get('correlation_score', 0)),
                float(position.get('days_since_last_action', 0)),
                float(position.get('price_change_24h', 0)),
                float(position.get('price_change_7d', 0)),
                float(position.get('market_stress_indicator', 0)),
                int(position.get('is_whale', False)),  # Large position indicator
                float(position.get('credit_score', 500))  # User credit score
            ]
            features.append(feature_vector)
            
        return np.array(features)
    
    def train(self, training_data):
        """Train the liquidation prediction model"""
        logger.info("Training liquidation prediction model...")
        
        X = self.prepare_features(training_data)
        y = np.array([int(d.get('was_liquidated', False)) for d in training_data])
        
        if len(X) < 10:
            logger.warning("Insufficient training data for liquidation model")
            return False
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        logger.info(f"Liquidation model - Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
        
        self.is_trained = True
        return True
    
    def predict_liquidation_probability(self, position_data):
        """Predict liquidation probability for a position"""
        if not self.is_trained:
            return 0.5  # Default probability if model not trained
            
        features = self.prepare_features([position_data])
        features_scaled = self.scaler.transform(features)
        
        probability = self.model.predict_proba(features_scaled)[0][1]  # Probability of liquidation
        return float(probability)
    
    def get_feature_importance(self):
        """Get feature importance for model interpretation"""
        if not self.is_trained:
            return {}
            
        feature_names = [
            'health_factor', 'collateral_value_usd', 'debt_value_usd', 'utilization_rate',
            'num_assets', 'largest_position_ratio', 'volatility_score', 'liquidity_score',
            'correlation_score', 'days_since_last_action', 'price_change_24h', 'price_change_7d',
            'market_stress_indicator', 'is_whale', 'credit_score'
        ]
        
        importance_dict = dict(zip(feature_names, self.model.feature_importances_))
        return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

class PricePredictionModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_price_features(self, market_data):
        """Extract features for price prediction"""
        features = []
        
        for data in market_data:
            feature_vector = [
                float(data.get('volume_24h', 0)),
                float(data.get('market_cap', 0)),
                float(data.get('liquidity_usd', 0)),
                float(data.get('volatility_7d', 0)),
                float(data.get('volatility_30d', 0)),
                float(data.get('rsi', 50)),  # Relative Strength Index
                float(data.get('moving_avg_7d', 0)),
                float(data.get('moving_avg_30d', 0)),
                float(data.get('price_change_1h', 0)),
                float(data.get('price_change_24h', 0)),
                float(data.get('price_change_7d', 0)),
                float(data.get('social_sentiment', 0)),
                float(data.get('fear_greed_index', 50)),
                float(data.get('network_activity', 0)),
                float(data.get('whale_movements', 0))
            ]
            features.append(feature_vector)
            
        return np.array(features)
    
    def train(self, training_data):
        """Train the price prediction model"""
        logger.info("Training price prediction model...")
        
        X = self.prepare_price_features(training_data)
        y = np.array([float(d.get('price_change_future', 0)) for d in training_data])
        
        if len(X) < 50:
            logger.warning("Insufficient training data for price prediction model")
            return False
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)
        
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        
        logger.info(f"Price model - Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        
        self.is_trained = True
        return True
    
    def predict_price_movement(self, market_data, time_horizon_hours=24):
        """Predict price movement for next time horizon"""
        if not self.is_trained:
            return 0.0
            
        features = self.prepare_price_features([market_data])
        features_scaled = self.scaler.transform(features)
        
        price_change = self.model.predict(features_scaled)[0]
        
        # Adjust for time horizon (model trained on 24h changes)
        adjusted_change = price_change * (time_horizon_hours / 24.0)
        
        return float(adjusted_change)

class RiskAssessmentModel:
    def __init__(self):
        self.correlation_matrix = {}
        self.volatility_models = {}
        
    def calculate_portfolio_var(self, positions, confidence_level=0.05):
        """Calculate Value at Risk for portfolio"""
        try:
            weights = []
            returns = []
            
            total_value = sum(float(p.get('value_usd', 0)) for p in positions)
            if total_value == 0:
                return 0.0
                
            for position in positions:
                weight = float(position.get('value_usd', 0)) / total_value
                return_volatility = float(position.get('volatility_30d', 0.1))
                
                weights.append(weight)
                returns.append(return_volatility)
            
            weights = np.array(weights)
            returns = np.array(returns)
            
            # Simplified VaR calculation (assumes normal distribution)
            portfolio_volatility = np.sqrt(np.dot(weights**2, returns**2))
            var = portfolio_volatility * 1.645  # 95% confidence level
            
            return float(var)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.1  # Default VaR
    
    def assess_concentration_risk(self, positions):
        """Assess concentration risk in portfolio"""
        total_value = sum(float(p.get('value_usd', 0)) for p in positions)
        if total_value == 0:
            return 0.0
            
        # Calculate Herfindahl-Hirschman Index
        hhi = sum((float(p.get('value_usd', 0)) / total_value) ** 2 for p in positions)
        
        # Convert to concentration risk score (0-1, higher is riskier)
        concentration_risk = min(1.0, hhi * 2)
        
        return float(concentration_risk)
    
    def calculate_correlation_risk(self, positions):
        """Calculate correlation risk between assets"""
        if len(positions) < 2:
            return 0.0
            
        # Simplified correlation risk calculation
        # In production, would use actual correlation matrices
        asset_types = [p.get('asset_type', 'crypto') for p in positions]
        unique_types = len(set(asset_types))
        
        # Higher diversity = lower correlation risk
        correlation_risk = max(0.0, 1.0 - (unique_types / len(positions)))
        
        return float(correlation_risk)

class YieldOptimizer:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
        
    def optimize_portfolio(self, available_assets, risk_tolerance, target_return=None):
        """Optimize portfolio allocation using simplified Markowitz approach"""
        try:
            # Extract expected returns and risks
            expected_returns = np.array([float(asset.get('expected_return', 0.05)) for asset in available_assets])
            risks = np.array([float(asset.get('volatility', 0.2)) for asset in available_assets])
            
            n_assets = len(available_assets)
            if n_assets == 0:
                return []
                
            # Equal weight as starting point
            weights = np.ones(n_assets) / n_assets
            
            # Risk budgeting approach based on risk tolerance
            risk_budget = np.ones(n_assets) / risks  # Inverse volatility weighting
            risk_budget = risk_budget / np.sum(risk_budget)
            
            # Adjust based on risk tolerance (0-1 scale)
            risk_adjustment = 1.0 - float(risk_tolerance)
            adjusted_weights = (1 - risk_adjustment) * weights + risk_adjustment * risk_budget
            
            # Normalize weights
            adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
            
            optimization_result = []
            for i, asset in enumerate(available_assets):
                optimization_result.append({
                    'asset': asset.get('address'),
                    'symbol': asset.get('symbol'),
                    'weight': float(adjusted_weights[i]),
                    'expected_return': float(expected_returns[i]),
                    'risk': float(risks[i]),
                    'sharpe_ratio': float((expected_returns[i] - self.risk_free_rate) / max(risks[i], 0.001))
                })
            
            # Sort by Sharpe ratio
            optimization_result.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return []

# Initialize ML models
liquidation_predictor = LiquidationPredictor()
price_predictor = PricePredictionModel()
risk_assessor = RiskAssessmentModel()
yield_optimizer = YieldOptimizer()

def get_cached_data(key, ttl=300):
    """Get data from Redis cache"""
    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.error(f"Redis get error: {e}")
    return None

def set_cached_data(key, data, ttl=300):
    """Set data in Redis cache"""
    try:
        redis_client.setex(key, ttl, json.dumps(data, default=str))
    except Exception as e:
        logger.error(f"Redis set error: {e}")

def load_training_data():
    """Load training data from database"""
    try:
        # Load liquidation events for training
        liquidation_events = list(db.liquidation_events.find().limit(1000))
        
        # Load market data
        market_data = list(db.market_data.find().limit(5000))
        
        # Load user positions
        user_positions = list(db.user_positions.find().limit(10000))
        
        return {
            'liquidations': liquidation_events,
            'market_data': market_data,
            'positions': user_positions
        }
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return {'liquidations': [], 'market_data': [], 'positions': []}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'models': {
            'liquidation': liquidation_predictor.is_trained,
            'price_prediction': price_predictor.is_trained
        }
    })

@app.route('/predict/liquidation', methods=['POST'])
def predict_liquidation():
    """Predict liquidation probability for a position"""
    try:
        data = request.json
        position_data = data.get('position', {})
        
        # Add cache key
        cache_key = f"liquidation_pred:{hash(str(sorted(position_data.items())))}"
        cached_result = get_cached_data(cache_key, ttl=60)
        
        if cached_result:
            return jsonify(cached_result)
        
        # Predict liquidation probability
        probability = liquidation_predictor.predict_liquidation_probability(position_data)
        
        # Calculate time to liquidation based on health factor trend
        health_factor = float(position_data.get('health_factor', 1.0))
        time_to_liquidation = estimate_time_to_liquidation(health_factor, probability)
        
        # Risk factors analysis
        risk_factors = analyze_risk_factors(position_data)
        
        result = {
            'liquidation_probability': probability,
            'time_to_liquidation_hours': time_to_liquidation,
            'risk_level': 'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.3 else 'LOW',
            'risk_factors': risk_factors,
            'recommendations': generate_liquidation_recommendations(probability, position_data)
        }
        
        set_cached_data(cache_key, result, ttl=60)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error predicting liquidation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/price', methods=['POST'])
def predict_price():
    """Predict price movement for an asset"""
    try:
        data = request.json
        market_data = data.get('market_data', {})
        time_horizon = data.get('time_horizon_hours', 24)
        
        cache_key = f"price_pred:{market_data.get('asset')}:{time_horizon}"
        cached_result = get_cached_data(cache_key, ttl=300)
        
        if cached_result:
            return jsonify(cached_result)
        
        # Predict price movement
        price_change = price_predictor.predict_price_movement(market_data, time_horizon)
        
        # Calculate confidence intervals
        confidence_intervals = calculate_confidence_intervals(price_change, market_data)
        
        result = {
            'predicted_price_change_percent': price_change * 100,
            'confidence_intervals': confidence_intervals,
            'prediction_quality': assess_prediction_quality(market_data),
            'time_horizon_hours': time_horizon,
            'factors': analyze_price_factors(market_data)
        }
        
        set_cached_data(cache_key, result, ttl=300)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error predicting price: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/assess/portfolio_risk', methods=['POST'])
def assess_portfolio_risk():
    """Assess comprehensive portfolio risk"""
    try:
        data = request.json
        positions = data.get('positions', [])
        
        cache_key = f"portfolio_risk:{hash(str(sorted([str(p) for p in positions])))}"
        cached_result = get_cached_data(cache_key, ttl=120)
        
        if cached_result:
            return jsonify(cached_result)
        
        # Calculate various risk metrics
        var_95 = risk_assessor.calculate_portfolio_var(positions, 0.05)
        var_99 = risk_assessor.calculate_portfolio_var(positions, 0.01)
        concentration_risk = risk_assessor.assess_concentration_risk(positions)
        correlation_risk = risk_assessor.calculate_correlation_risk(positions)
        
        # Overall risk score
        overall_risk = calculate_overall_risk_score(var_95, concentration_risk, correlation_risk)
        
        result = {
            'value_at_risk_95': var_95,
            'value_at_risk_99': var_99,
            'concentration_risk': concentration_risk,
            'correlation_risk': correlation_risk,
            'overall_risk_score': overall_risk,
            'risk_breakdown': {
                'market_risk': var_95 * 0.6,
                'concentration_risk': concentration_risk * 0.25,
                'correlation_risk': correlation_risk * 0.15
            },
            'recommendations': generate_risk_recommendations(var_95, concentration_risk, correlation_risk)
        }
        
        set_cached_data(cache_key, result, ttl=120)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error assessing portfolio risk: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/optimize/yield', methods=['POST'])
def optimize_yield():
    """Optimize portfolio for yield"""
    try:
        data = request.json
        available_assets = data.get('available_assets', [])
        risk_tolerance = data.get('risk_tolerance', 0.5)
        current_portfolio = data.get('current_portfolio', [])
        
        cache_key = f"yield_opt:{hash(str(available_assets))}:{risk_tolerance}"
        cached_result = get_cached_data(cache_key, ttl=600)
        
        if cached_result:
            return jsonify(cached_result)
        
        # Optimize portfolio
        optimized_allocation = yield_optimizer.optimize_portfolio(
            available_assets, risk_tolerance
        )
        
        # Calculate rebalancing suggestions
        rebalancing_suggestions = calculate_rebalancing_suggestions(
            current_portfolio, optimized_allocation
        )
        
        # Estimate returns and risks
        portfolio_metrics = calculate_portfolio_metrics(optimized_allocation)
        
        result = {
            'optimized_allocation': optimized_allocation,
            'rebalancing_suggestions': rebalancing_suggestions,
            'expected_annual_return': portfolio_metrics['expected_return'],
            'estimated_risk': portfolio_metrics['risk'],
            'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
            'diversification_score': portfolio_metrics['diversification_score']
        }
        
        set_cached_data(cache_key, result, ttl=600)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error optimizing yield: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train/models', methods=['POST'])
def train_models():
    """Train ML models with latest data"""
    try:
        # Load training data
        training_data = load_training_data()
        
        # Train liquidation model
        liquidation_success = liquidation_predictor.train(training_data['positions'])
        
        # Train price prediction model
        price_success = price_predictor.train(training_data['market_data'])
        
        # Save models
        if liquidation_success:
            joblib.dump(liquidation_predictor, 'models/liquidation_model.pkl')
        if price_success:
            joblib.dump(price_predictor, 'models/price_model.pkl')
        
        result = {
            'success': True,
            'models_trained': {
                'liquidation': liquidation_success,
                'price_prediction': price_success
            },
            'training_data_size': {
                'positions': len(training_data['positions']),
                'market_data': len(training_data['market_data']),
                'liquidations': len(training_data['liquidations'])
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return jsonify({'error': str(e)}), 500

# Utility functions
def estimate_time_to_liquidation(health_factor, probability):
    """Estimate time to liquidation based on health factor and probability"""
    if health_factor >= 1.5:
        return 168  # 1 week
    elif health_factor >= 1.2:
        return 24   # 1 day
    elif health_factor >= 1.1:
        return 4    # 4 hours
    elif health_factor >= 1.05:
        return 1    # 1 hour
    else:
        return 0.25 # 15 minutes

def analyze_risk_factors(position_data):
    """Analyze specific risk factors for a position"""
    factors = []
    
    health_factor = float(position_data.get('health_factor', 1.0))
    if health_factor < 1.2:
        factors.append({
            'factor': 'Low Health Factor',
            'severity': 'HIGH',
            'description': f'Health factor of {health_factor:.3f} is dangerously low'
        })
    
    utilization = float(position_data.get('utilization_rate', 0))
    if utilization > 0.8:
        factors.append({
            'factor': 'High Utilization',
            'severity': 'MEDIUM',
            'description': f'Utilization rate of {utilization:.1%} is quite high'
        })
    
    concentration = float(position_data.get('largest_position_ratio', 0))
    if concentration > 0.5:
        factors.append({
            'factor': 'Concentration Risk',
            'severity': 'MEDIUM',
            'description': f'Largest position represents {concentration:.1%} of portfolio'
        })
    
    return factors

def generate_liquidation_recommendations(probability, position_data):
    """Generate recommendations to avoid liquidation"""
    recommendations = []
    
    if probability > 0.5:
        recommendations.append({
            'action': 'Add Collateral',
            'priority': 'HIGH',
            'description': 'Deposit additional collateral to improve health factor'
        })
        
        recommendations.append({
            'action': 'Partial Repayment',
            'priority': 'HIGH',
            'description': 'Repay part of your debt to reduce liquidation risk'
        })
    
    if probability > 0.3:
        recommendations.append({
            'action': 'Diversify Collateral',
            'priority': 'MEDIUM',
            'description': 'Use multiple types of collateral to reduce correlation risk'
        })
    
    return recommendations

def calculate_confidence_intervals(predicted_change, market_data):
    """Calculate confidence intervals for price predictions"""
    volatility = float(market_data.get('volatility_7d', 0.2))
    
    # Simple confidence intervals based on volatility
    confidence_95 = predicted_change * (1 + 1.96 * volatility)
    confidence_5 = predicted_change * (1 - 1.96 * volatility)
    
    return {
        '95th_percentile': confidence_95,
        '5th_percentile': confidence_5,
        'volatility_adjusted': volatility
    }

def assess_prediction_quality(market_data):
    """Assess the quality/confidence of predictions"""
    liquidity = float(market_data.get('liquidity_usd', 0))
    volatility = float(market_data.get('volatility_7d', 1.0))
    
    # Higher liquidity and lower volatility = better prediction quality
    liquidity_score = min(1.0, liquidity / 10000000)  # Normalize to $10M
    volatility_score = max(0.0, 1.0 - volatility)
    
    quality_score = (liquidity_score + volatility_score) / 2
    
    if quality_score > 0.8:
        return 'HIGH'
    elif quality_score > 0.5:
        return 'MEDIUM'
    else:
        return 'LOW'

def analyze_price_factors(market_data):
    """Analyze factors affecting price prediction"""
    factors = {
        'volume_trend': 'NEUTRAL',
        'volatility_level': 'MEDIUM',
        'liquidity_condition': 'GOOD',
        'market_sentiment': 'NEUTRAL'
    }
    
    volume_24h = float(market_data.get('volume_24h', 0))
    if volume_24h > 1000000:  # $1M+
        factors['volume_trend'] = 'HIGH'
    elif volume_24h < 100000:  # $100K
        factors['volume_trend'] = 'LOW'
    
    volatility = float(market_data.get('volatility_7d', 0))
    if volatility > 0.5:
        factors['volatility_level'] = 'HIGH'
    elif volatility < 0.1:
        factors['volatility_level'] = 'LOW'
    
    return factors

def calculate_overall_risk_score(var, concentration, correlation):
    """Calculate overall portfolio risk score"""
    # Weighted combination of risk factors
    weights = {'var': 0.5, 'concentration': 0.3, 'correlation': 0.2}
    
    normalized_var = min(1.0, var / 0.5)  # Normalize VaR to 50%
    
    overall_risk = (
        normalized_var * weights['var'] +
        concentration * weights['concentration'] +
        correlation * weights['correlation']
    )
    
    return min(1.0, overall_risk)

def generate_risk_recommendations(var, concentration, correlation):
    """Generate risk management recommendations"""
    recommendations = []
    
    if var > 0.3:
        recommendations.append({
            'type': 'RISK_REDUCTION',
            'message': 'Consider reducing position sizes to lower portfolio volatility'
        })
    
    if concentration > 0.6:
        recommendations.append({
            'type': 'DIVERSIFICATION',
            'message': 'Diversify across more assets to reduce concentration risk'
        })
    
    if correlation > 0.7:
        recommendations.append({
            'type': 'CORRELATION',
            'message': 'Add uncorrelated assets to reduce portfolio correlation risk'
        })
    
    return recommendations

def calculate_rebalancing_suggestions(current, optimized):
    """Calculate rebalancing suggestions"""
    suggestions = []
    
    current_dict = {asset['asset']: float(asset['weight']) for asset in current}
    
    for asset in optimized:
        asset_address = asset['asset']
        optimal_weight = asset['weight']
        current_weight = current_dict.get(asset_address, 0.0)
        
        difference = optimal_weight - current_weight
        
        if abs(difference) > 0.05:  # 5% threshold
            suggestions.append({
                'asset': asset_address,
                'symbol': asset['symbol'],
                'action': 'INCREASE' if difference > 0 else 'DECREASE',
                'current_weight': current_weight,
                'target_weight': optimal_weight,
                'adjustment': abs(difference)
            })
    
    return suggestions

def calculate_portfolio_metrics(allocation):
    """Calculate portfolio performance metrics"""
    if not allocation:
        return {
            'expected_return': 0.0,
            'risk': 0.0,
            'sharpe_ratio': 0.0,
            'diversification_score': 0.0
        }
    
    weights = np.array([asset['weight'] for asset in allocation])
    returns = np.array([asset['expected_return'] for asset in allocation])
    risks = np.array([asset['risk'] for asset in allocation])
    
    # Portfolio expected return
    portfolio_return = np.dot(weights, returns)
    
    # Portfolio risk (simplified - assumes no correlation)
    portfolio_risk = np.sqrt(np.dot(weights**2, risks**2))
    
    # Sharpe ratio
    risk_free_rate = 0.02
    sharpe_ratio = (portfolio_return - risk_free_rate) / max(portfolio_risk, 0.001)
    
    # Diversification score (entropy-based)
    diversification_score = -np.sum(weights * np.log(weights + 1e-10))
    diversification_score = min(1.0, diversification_score / np.log(len(weights)))
    
    return {
        'expected_return': float(portfolio_return),
        'risk': float(portfolio_risk),
        'sharpe_ratio': float(sharpe_ratio),
        'diversification_score': float(diversification_score)
    }

# Load pre-trained models if they exist
def load_pretrained_models():
    """Load pre-trained models from disk"""
    try:
        if os.path.exists('models/liquidation_model.pkl'):
            global liquidation_predictor
            liquidation_predictor = joblib.load('models/liquidation_model.pkl')
            logger.info("Loaded pre-trained liquidation model")
        
        if os.path.exists('models/price_model.pkl'):
            global price_predictor
            price_predictor = joblib.load('models/price_model.pkl')
            logger.info("Loaded pre-trained price prediction model")
            
    except Exception as e:
        logger.error(f"Error loading pre-trained models: {e}")

if __name__ == '__main__':
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load pre-trained models
    load_pretrained_models()
    
    # Start Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5001)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
      )
