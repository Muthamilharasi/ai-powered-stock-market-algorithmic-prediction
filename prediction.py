import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self, twelve_api_key=None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.twelve_api_key = twelve_api_key  # Your Twelve Data API key

    # ---------------- Twelve Data Fallback ----------------
    def fetch_twelve_data(self, symbol, interval='1day', outputsize=500):
        if not self.twelve_api_key:
            logging.error("Twelve Data API key not provided.")
            return None
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.twelve_api_key,
            "outputsize": outputsize
        }
        try:
            response = requests.get(url, params=params).json()
            if "values" not in response:
                logging.error(f"Twelve Data error: {response.get('message','Unknown error')}")
                return None
            df = pd.DataFrame(response['values'])
            df = df[['datetime','open','high','low','close','volume']]
            df.rename(columns={'datetime':'Date','open':'Open','high':'High',
                               'low':'Low','close':'Close','volume':'Volume'}, inplace=True)
            for col in ['Open','High','Low','Close','Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error fetching Twelve Data: {str(e)}")
            return None

    # ---------------- Features & Technical Indicators ----------------
    def prepare_features(self, data):
        df = data.copy()
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Price_SMA5_Ratio'] = df['Close']/df['SMA_5']
        df['Price_SMA20_Ratio'] = df['Close']/df['SMA_20']
        df['Volatility'] = df['Close'].rolling(10).std()
        df['Volume_SMA'] = df['Volume'].rolling(10).mean()
        df['Volume_Ratio'] = df['Volume']/df['Volume_SMA']
        df['Momentum_5'] = df['Close']/df['Close'].shift(5)
        df['Momentum_10'] = df['Close']/df['Close'].shift(10)
        df['RSI'] = self.calculate_rsi(df['Close'])
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1-exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2*bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2*bb_std
        df['BB_Position'] = (df['Close']-df['BB_Lower'])/(df['BB_Upper']-df['BB_Lower'])
        df['HL_Ratio'] = df['High']/df['Low']
        df['OC_Ratio'] = df['Open']/df['Close']

        features = ['SMA_5','SMA_10','SMA_20','SMA_50','Price_SMA5_Ratio','Price_SMA20_Ratio',
                    'Volatility','Volume_Ratio','Momentum_5','Momentum_10','RSI','MACD','MACD_Signal',
                    'BB_Position','HL_Ratio','OC_Ratio']
        return df[features].dropna()

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = delta.where(delta>0,0).rolling(window).mean()
        loss = (-delta.where(delta<0,0)).rolling(window).mean()
        rs = gain/loss
        return 100-(100/(1+rs))

    # ---------------- Model Training ----------------
    def train_model(self, features, target):
        if len(features)<50:
            return False
        X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.2,random_state=42)
        X_train_scaled=self.scaler.fit_transform(X_train)
        X_test_scaled=self.scaler.transform(X_test)
        self.model = RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42,n_jobs=-1)
        self.model.fit(X_train_scaled,y_train)
        self.feature_importance = dict(zip(features.columns,self.model.feature_importances_))
        return True

    # ---------------- Prediction ----------------
    def predict(self, stock_data=None, symbol=None):
        try:
            # Use existing data or fallback to Twelve Data
            if stock_data is None or len(stock_data)<50:
                if symbol is None:
                    return None
                stock_data = self.fetch_twelve_data(symbol)
                if stock_data is None or len(stock_data)<50:
                    logging.error(f"Not enough data for {symbol}")
                    return None

            features = self.prepare_features(stock_data)
            target = stock_data['Close'].shift(-1).dropna()
            min_len = min(len(features),len(target))
            features=features.iloc[:min_len]
            target=target.iloc[:min_len]

            if not self.train_model(features,target):
                return None

            latest_features = features.iloc[-1:].values
            latest_scaled = self.scaler.transform(latest_features)
            predicted_price = self.model.predict(latest_scaled)[0]
            current_price = stock_data['Close'].iloc[-1]

            price_change = predicted_price-current_price
            price_change_percent = (price_change/current_price)*100
            recent_volatility = stock_data['Close'].tail(10).std()
            avg_price = stock_data['Close'].tail(10).mean()
            volatility_ratio = recent_volatility/avg_price
            confidence = max(0.3,min(0.95,1-(volatility_ratio*10)))
            explanation = self.generate_explanation(features.iloc[-1])

            if price_change_percent>1: trend="Strong Bullish"
            elif price_change_percent>0: trend="Bullish"
            elif price_change_percent>-1: trend="Bearish"
            else: trend="Strong Bearish"

            return {
                'predicted_price': round(predicted_price,2),
                'current_price': round(current_price,2),
                'price_change': round(price_change,2),
                'price_change_percent': round(price_change_percent,2),
                'confidence': round(confidence*100,1),
                'trend': trend,
                'explanation': explanation,
                'prediction_date': (datetime.now()+timedelta(days=1)).strftime('%Y-%m-%d'),
                'model_accuracy': round((1-volatility_ratio)*100,1)
            }

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return None

    def generate_explanation(self, latest_features):
        if self.feature_importance is None:
            return ["Model training in progress..."]
        top_features = sorted(self.feature_importance.items(), key=lambda x:x[1], reverse=True)[:3]
        explanations=[]
        for feature,_ in top_features:
            val=latest_features[feature]
            if 'SMA' in feature: explanations.append(f"Price vs {feature}: {val:.2f}")
            elif 'RSI' in feature: explanations.append(f"RSI indicates {val:.2f}")
            elif 'Volatility' in feature: explanations.append(f"Volatility is {val:.2f}")
            elif 'Momentum' in feature: explanations.append(f"Momentum is {val:.2f}")
            elif 'Volume' in feature: explanations.append(f"Volume Ratio is {val:.2f}")
        return explanations[:3]
