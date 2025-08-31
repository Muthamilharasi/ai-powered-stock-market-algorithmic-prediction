import random
import time
import math
from datetime import datetime

class StockSimulator:
    def __init__(self):
        # Initialize with popular stocks and their base prices
        self.stocks = {
            'AAPL': {'price': 175.00, 'base_price': 175.00, 'volatility': 0.02},
            'GOOGL': {'price': 135.00, 'base_price': 135.00, 'volatility': 0.025},
            'MSFT': {'price': 340.00, 'base_price': 340.00, 'volatility': 0.02},
            'AMZN': {'price': 145.00, 'base_price': 145.00, 'volatility': 0.03},
            'TSLA': {'price': 240.00, 'base_price': 240.00, 'volatility': 0.05},
            'NVDA': {'price': 450.00, 'base_price': 450.00, 'volatility': 0.04},
            'META': {'price': 320.00, 'base_price': 320.00, 'volatility': 0.035},
            'NFLX': {'price': 420.00, 'base_price': 420.00, 'volatility': 0.03},
            'ADBE': {'price': 530.00, 'base_price': 530.00, 'volatility': 0.025},
            'CRM': {'price': 210.00, 'base_price': 210.00, 'volatility': 0.03}
        }
        
        # Store price history for charts
        self.price_history = {}
        for symbol in self.stocks:
            self.price_history[symbol] = []
        
        self.last_update = time.time()
    
    def update_prices(self):
        """Update stock prices with realistic movements"""
        current_time = time.time()
        
        # Market hours simulation (more volatility during trading hours)
        now = datetime.now()
        hour = now.hour
        
        # Simulate market hours (9:30 AM - 4:00 PM EST)
        market_open = 9.5 <= hour <= 16
        volatility_multiplier = 1.0 if market_open else 0.3
        
        for symbol, data in self.stocks.items():
            # Generate price movement using geometric Brownian motion
            dt = 1.0 / 3600  # 1 second time step in hours
            volatility = data['volatility'] * volatility_multiplier
            
            # Random walk with mean reversion
            random_component = random.gauss(0, 1)
            
            # Mean reversion factor (tendency to return to base price)
            mean_reversion = 0.001 * (data['base_price'] - data['price']) / data['base_price']
            
            # Market trend factor (small upward bias)
            trend = 0.0001
            
            # Calculate price change
            price_change = data['price'] * (
                (trend + mean_reversion) * dt + 
                volatility * math.sqrt(dt) * random_component
            )
            
            # Update price
            new_price = data['price'] + price_change
            
            # Ensure price doesn't go negative or too extreme
            min_price = data['base_price'] * 0.5
            max_price = data['base_price'] * 2.0
            new_price = max(min_price, min(max_price, new_price))
            
            data['price'] = round(new_price, 2)
            
            # Store price history (keep last 100 points)
            timestamp = int(current_time * 1000)
            self.price_history[symbol].append({
                'timestamp': timestamp,
                'price': data['price']
            })
            
            # Keep only last 100 data points
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol].pop(0)
        
        self.last_update = current_time
    
    def get_price(self, symbol):
        """Get current price for a symbol"""
        return self.stocks.get(symbol, {}).get('price', 0.0)
    
    def get_all_prices(self):
        """Get all current prices"""
        return {
            symbol: {
                'price': data['price'],
                'change': round(data['price'] - data['base_price'], 2),
                'change_percent': round(((data['price'] - data['base_price']) / data['base_price']) * 100, 2),
                'history': self.price_history.get(symbol, [])
            }
            for symbol, data in self.stocks.items()
        }
    
    def get_price_history(self, symbol, limit=50):
        """Get price history for a symbol"""
        return self.price_history.get(symbol, [])[-limit:]
