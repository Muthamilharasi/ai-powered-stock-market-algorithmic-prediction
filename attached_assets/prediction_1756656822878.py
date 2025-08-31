import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def prepare_features(self, data):
        """Prepare features for machine learning model"""
        df = data.copy()
        
        # Calculate technical indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Price ratios
        df['Price_SMA5_Ratio'] = df['Close'] / df['SMA_5']
        df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10)
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # High-Low ratios
        df['HL_Ratio'] = df['High'] / df['Low']
        df['OC_Ratio'] = df['Open'] / df['Close']
        
        # Select features for prediction
        feature_columns = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'Price_SMA5_Ratio', 'Price_SMA20_Ratio',
            'Volatility', 'Volume_Ratio',
            'Momentum_5', 'Momentum_10',
            'RSI', 'MACD', 'MACD_Signal',
            'BB_Position', 'HL_Ratio', 'OC_Ratio'
        ]
        
        return df[feature_columns].dropna()
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_model(self, features, target):
        """Train the prediction model"""
        if len(features) < 50:  # Need minimum data for training
            return False
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            features.columns,
            self.model.feature_importances_
        ))
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        logging.info(f"Model trained. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        return True
    
    def predict(self, stock_data):
        """Generate prediction for stock data"""
        try:
            # Prepare features
            features = self.prepare_features(stock_data)
            
            if len(features) < 50:
                return None
            
            # Prepare target (next day's closing price)
            target = stock_data['Close'].shift(-1).dropna()
            
            # Align features and target
            min_length = min(len(features), len(target))
            features = features.iloc[:min_length]
            target = target.iloc[:min_length]
            
            # Train model
            if not self.train_model(features, target):
                return None
            
            # Make prediction for next day
            latest_features = features.iloc[-1:].values
            latest_features_scaled = self.scaler.transform(latest_features)
            
            predicted_price = self.model.predict(latest_features_scaled)[0]
            current_price = stock_data['Close'].iloc[-1]
            
            # Calculate prediction metrics
            price_change = predicted_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            # Generate confidence score based on recent volatility
            recent_volatility = stock_data['Close'].tail(10).std()
            avg_price = stock_data['Close'].tail(10).mean()
            volatility_ratio = recent_volatility / avg_price
            confidence = max(0.3, min(0.95, 1 - (volatility_ratio * 10)))
            
            # Generate explanation based on feature importance
            explanation = self.generate_explanation(features.iloc[-1])
            
            # Determine trend
            if price_change_percent > 1:
                trend = "Strong Bullish"
            elif price_change_percent > 0:
                trend = "Bullish"
            elif price_change_percent > -1:
                trend = "Bearish"
            else:
                trend = "Strong Bearish"
            
            return {
                'predicted_price': round(predicted_price, 2),
                'current_price': round(current_price, 2),
                'price_change': round(price_change, 2),
                'price_change_percent': round(price_change_percent, 2),
                'confidence': round(confidence * 100, 1),
                'trend': trend,
                'explanation': explanation,
                'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'model_accuracy': round((1 - volatility_ratio) * 100, 1)
            }
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return None
    
    def generate_explanation(self, latest_features):
        """Generate explainable AI reasoning for the prediction"""
        if self.feature_importance is None:
            return "Model training in progress..."
        
        # Get top 3 most important features
        top_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        explanations = []
        
        for feature, importance in top_features:
            feature_value = latest_features[feature]
            
            if 'SMA' in feature:
                if feature_value > 1:
                    explanations.append(f"Price is above {feature.replace('_', ' ')} (bullish indicator)")
                else:
                    explanations.append(f"Price is below {feature.replace('_', ' ')} (bearish indicator)")
            
            elif 'RSI' in feature:
                if feature_value > 70:
                    explanations.append("RSI indicates overbought conditions (potential reversal)")
                elif feature_value < 30:
                    explanations.append("RSI indicates oversold conditions (potential upside)")
                else:
                    explanations.append("RSI shows neutral momentum")
            
            elif 'Volatility' in feature:
                if feature_value > latest_features.std():
                    explanations.append("High volatility suggests increased uncertainty")
                else:
                    explanations.append("Low volatility suggests stable price movement")
            
            elif 'Volume' in feature:
                if feature_value > 1:
                    explanations.append("Above-average volume supports price movement")
                else:
                    explanations.append("Below-average volume indicates weak conviction")
            
            elif 'Momentum' in feature:
                if feature_value > 1:
                    explanations.append("Positive momentum suggests continued upward movement")
                else:
                    explanations.append("Negative momentum suggests downward pressure")
        
        return explanations[:3]  # Return top 3 explanations
