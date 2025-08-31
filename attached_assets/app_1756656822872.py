import os
import re
import time
import json
import random
import logging
import threading
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
from telegram import Bot
from sqlalchemy.orm import DeclarativeBase

# Local imports
from models import User, Portfolio, Trade, Position, Alert, db,Transaction
from news import analyze_news
from prediction import StockPredictor
from paper_trading import PaperTradingEngine
from stock_simulator import StockSimulator

# ------------------ CONFIG ------------------
load_dotenv()
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "supersecret")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_recycle": 300, "pool_pre_ping": True}
db.init_app(app)

# Extensions
bcrypt = Bcrypt(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
telegram_bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))

# Trading engines
trading_engine = PaperTradingEngine(db)
stock_simulator = StockSimulator()
predictor = StockPredictor()

# ------------------ HELPERS ------------------
def send_telegram_alert(message):
    try:
        if os.getenv("TELEGRAM_CHAT_ID"):
            telegram_bot.send_message(chat_id=os.getenv("TELEGRAM_CHAT_ID"), text=message)
    except Exception as e:
        logging.error(f"Telegram alert failed: {e}")

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def check_alerts():
    alerts = Alert.query.filter_by(status="Active").all()
    for alert in alerts:
        try:
            match = re.match(r"Price\s*([<>]=?|==)\s*(\d+\.?\d*)", alert.condition)
            if not match:
                continue
            operator, value = match.groups()
            value = float(value)

            data = yf.Ticker(alert.symbol).history(period="1d")
            if data.empty:
                continue
            current_price = float(data['Close'].iloc[-1])

            if eval(f"{current_price} {operator} {value}"):
                user = User.query.get(alert.user_id)
                if user and user.alerts_enabled:
                    send_telegram_alert(f"Alert Triggered: {alert.symbol} {alert.condition} (Current: {current_price})")
                alert.status = "Triggered"
                db.session.commit()
        except Exception as e:
            logging.error(f"Error checking alert: {e}")

# ------------------ AUTH ------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username, password = request.form['username'], request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash("Welcome back!", "success")
            return redirect(url_for('dashboard'))
        flash("Invalid credentials.", "danger")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
            return redirect(url_for('register'))
        db.session.add(User(username=username, password=password))
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')
@app.route("/profile",methods=['GET', 'POST'])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))   # redirect if not logged in

    user = User.query.get(session["user_id"])
    transactions = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.timestamp.desc()).all()

    return render_template("profile.html", user=user, transactions=transactions)


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# ------------------ DASHBOARD ------------------
@app.route('/')
def home():
    return redirect(url_for('dashboard')) if 'user_id' in session else redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: 
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    watchlist_symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]
    watchlist = []
    for sym in watchlist_symbols:
        data = yf.Ticker(sym).history(period="2d")
        if not data.empty:
            price = round(data['Close'].iloc[-1], 2)
            prev = round(data['Close'].iloc[-2], 2) if len(data) > 1 else price
            change = round(((price - prev) / prev) * 100, 2)
            watchlist.append({"symbol": sym, "price": price, "change": change})
    return render_template('dashboard.html', user=user, watchlist=watchlist)

# ------------------ PREDICTION ------------------
predictor = StockPredictor()

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    """Main page with stock prediction interface"""
    return render_template('prediction.html')

def get_stock_symbol_variations(symbol):
    """Get different stock symbol variations for international exchanges"""
    symbol = symbol.upper().strip()
    
    # Common stock symbol mappings for international companies
    # Using verified symbols that work in Yahoo Finance
    symbol_mappings = {
        # Indian Companies (ADRs available in US markets)
        'TCS': ['INFY', 'TCS.NS'],  # Using Infosys ADR as TCS is not available as ADR
        'TATA': ['TTM', 'TATAMOTORS.NS'],  # Tata Motors ADR
        'RELIANCE': ['RELIANCE.NS'],  # No ADR, only Indian exchange
        'INFOSYS': ['INFY', 'INFY.NS'],  # Infosys ADR
        'WIPRO': ['WIT', 'WIPRO.NS'],  # Wipro ADR
        'HDFC': ['HDB', 'HDFCBANK.NS'],  # HDFC Bank ADR
        'ICICI': ['IBN', 'ICICIBANK.NS'],  # ICICI Bank ADR
        'ITC': ['ITC.NS'],  # Only on Indian exchanges
        'SBI': ['SBIN.NS'],  # Only on Indian exchanges
        'LT': ['LT.NS'],  # Larsen & Toubro
        'MARUTI': ['MARUTI.NS'],  # Maruti Suzuki
        'BAJAJ': ['BAJFINANCE.NS'],  # Bajaj Finance
        'ASIAN': ['ASIANPAINT.NS'],  # Asian Paints
        
        # Global Companies
        'SAMSUNG': ['005930.KS'],  # Samsung Electronics (Korea)
        'MICROSOFT': ['MSFT'],  # Microsoft
        'XOM': ['XOM'],  # ExxonMobil
        'BABA': ['BABA'],  # Alibaba ADR
        'TSM': ['TSM'],  # Taiwan Semiconductor ADR
        'TOYOTA': ['TM'],  # Toyota ADR
        'NESTLE': ['NSRGY'],  # Nestle ADR
        'UNILEVER': ['UL'],  # Unilever ADR
        'ROCHE': ['RHHBY'],  # Roche ADR
        'NOVARTIS': ['NVS'],  # Novartis ADR
        'SAP': ['SAP'],  # SAP
        'ASML': ['ASML'],  # ASML
        'SHELL': ['SHEL'],  # Shell
        'BP': ['BP'],  # BP
        'VODAFONE': ['VOD'],  # Vodafone
        
        # Additional popular companies
        'APPLE': ['AAPL'],
        'GOOGLE': ['GOOGL', 'GOOG'],
        'AMAZON': ['AMZN'],
        'TESLA': ['TSLA'],
        'META': ['META'],
        'NVIDIA': ['NVDA'],
        'NETFLIX': ['NFLX'],
        'ADOBE': ['ADBE'],
        'SALESFORCE': ['CRM'],
        'INTEL': ['INTC'],
        'AMD': ['AMD'],
        'ORACLE': ['ORCL'],
        'IBM': ['IBM'],
        'CISCO': ['CSCO'],
        'PAYPAL': ['PYPL'],
        'VISA': ['V'],
        'MASTERCARD': ['MA'],
        'JPMORGAN': ['JPM'],
        'BERKSHIRE': ['BRK-B', 'BRK-A'],
        'COCA': ['KO'],
        'DISNEY': ['DIS'],
        'WALMART': ['WMT'],
        'JOHNSON': ['JNJ'],
        'PROCTER': ['PG']
    }
    
    # If symbol is in mappings, return all variations
    if symbol in symbol_mappings:
        return symbol_mappings[symbol]
    
    # For other symbols, try common exchange suffixes
    variations = [symbol]
    
    # Add common exchange suffixes if not already present
    if '.' not in symbol:
        variations.extend([
            f"{symbol}.NS",  # NSE India
            f"{symbol}.BO",  # BSE India
            f"{symbol}.L",   # London
            f"{symbol}.DE",  # Germany
            f"{symbol}.PA",  # Paris
            f"{symbol}.TO",  # Toronto
            f"{symbol}.AS",  # Amsterdam
            f"{symbol}.SW",  # Switzerland
            f"{symbol}.HK",  # Hong Kong
            f"{symbol}.T",   # Tokyo
            f"{symbol}.KS"   # Korea
        ])
    
    return variations

@app.route('/api/stock-data/<symbol>')
def get_stock_data(symbol):
    """Fetch real-time stock data for a given symbol"""
    try:
        # Validate symbol
        if not symbol or len(symbol) > 20:  # Increased length for exchange suffixes
            return jsonify({'error': 'Invalid stock symbol'}), 400
        
        symbol = symbol.upper().strip()
        
        # Get symbol variations to try
        symbol_variations = get_stock_symbol_variations(symbol)
        
        stock_data = None
        working_symbol = None
        
        # Try each symbol variation until we find one that works
        for try_symbol in symbol_variations:
            try:
                stock = yf.Ticker(try_symbol)
                hist = stock.history(period="1y")
                
                if not hist.empty and len(hist) > 10:  # Ensure we have enough data
                    stock_data = stock
                    working_symbol = try_symbol
                    break
                    
            except Exception as e:
                logging.debug(f"Failed to fetch data for {try_symbol}: {str(e)}")
                continue
        
        if stock_data is None:
            return jsonify({'error': f'No data found for {symbol}. Please check the symbol or try the full exchange symbol (e.g., TCS.NS for Indian stocks)'}), 404
        
        # Use the working symbol for the rest of the function
        symbol = working_symbol
        stock = stock_data
        
        # Get historical data (1 year) - we already fetched this above
        hist = stock.history(period="1y")
        
        if hist.empty:
            return jsonify({'error': f'No data found for symbol {symbol}'}), 404
        
        # Get current stock info
        info = stock.info
        
        # Prepare chart data
        chart_data = []
        for date, row in hist.iterrows():
            chart_data.append({
                'x': date.strftime('%Y-%m-%d'),
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2),
                'volume': int(row['Volume'])
            })
        
        # Get current price
        current_price = round(float(hist['Close'].iloc[-1]), 2)
        prev_close = round(float(hist['Close'].iloc[-2]), 2)
        price_change = round(current_price - prev_close, 2)
        price_change_percent = round((price_change / prev_close) * 100, 2)
        
        # Calculate basic technical indicators
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        
        # Prepare response
        response_data = {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'current_price': current_price,
            'price_change': price_change,
            'price_change_percent': price_change_percent,
            'chart_data': chart_data,
            'technical_indicators': {
                'sma_20': round(float(hist['SMA_20'].iloc[-1]), 2) if not pd.isna(hist['SMA_20'].iloc[-1]) else None,
                'sma_50': round(float(hist['SMA_50'].iloc[-1]), 2) if not pd.isna(hist['SMA_50'].iloc[-1]) else None,
                'rsi': round(float(hist['RSI'].iloc[-1]), 2) if not pd.isna(hist['RSI'].iloc[-1]) else None,
                'volume': int(hist['Volume'].iloc[-1])
            },
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'dividend_yield': info.get('dividendYield')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Error fetching stock data for {symbol}: {str(e)}")
        return jsonify({'error': f'Failed to fetch data for {symbol}. Please check the symbol and try again.'}), 500

@app.route('/api/predict/<symbol>')
def predict_stock(symbol):
    """Generate AI prediction for a stock symbol"""
    try:
        symbol = symbol.upper().strip()
        
        # Get symbol variations and find working symbol
        symbol_variations = get_stock_symbol_variations(symbol)
        working_symbol = None
        
        for try_symbol in symbol_variations:
            try:
                stock = yf.Ticker(try_symbol)
                hist = stock.history(period="2y")  # More data for better prediction
                
                if not hist.empty and len(hist) > 50:  # Need enough data for prediction
                    working_symbol = try_symbol
                    break
                    
            except Exception as e:
                logging.debug(f"Failed to fetch prediction data for {try_symbol}: {str(e)}")
                continue
        
        if working_symbol is None:
            return jsonify({'error': f'Insufficient data for prediction. Please check the symbol.'}), 404
        
        # Use the working symbol
        stock = yf.Ticker(working_symbol)
        hist = stock.history(period="2y")
        
        # Generate prediction
        prediction_result = predictor.predict(hist)
        
        if prediction_result is None:
            return jsonify({'error': 'Unable to generate prediction. Insufficient data.'}), 400
        
        return jsonify(prediction_result)
        
    except Exception as e:
        logging.error(f"Error generating prediction for {symbol}: {str(e)}")
        return jsonify({'error': f'Failed to generate prediction for {symbol}'}), 500

@app.route('/api/supported-companies')
def get_supported_companies():
    """Get list of supported company symbols"""
    symbol_mappings = {
        # Indian Companies (ADRs available in US markets)
        'TCS': 'Tata Consultancy Services (use INFY for ADR)',
        'TATA': 'Tata Motors (use TTM)',
        'RELIANCE': 'Reliance Industries (RELIANCE.NS)',
        'INFOSYS': 'Infosys (use INFY)',
        'WIPRO': 'Wipro (use WIT)',
        'HDFC': 'HDFC Bank (use HDB)',
        'ICICI': 'ICICI Bank (use IBN)',
        
        # Global Companies
        'MICROSOFT': 'Microsoft (use MSFT)',
        'APPLE': 'Apple (use AAPL)',
        'GOOGLE': 'Google/Alphabet (use GOOGL)',
        'AMAZON': 'Amazon (use AMZN)',
        'TESLA': 'Tesla (use TSLA)',
        'META': 'Meta/Facebook (use META)',
        'NVIDIA': 'NVIDIA (use NVDA)',
        'SAMSUNG': 'Samsung Electronics (use 005930.KS)',
        'TOYOTA': 'Toyota (use TM)',
        'XOM': 'ExxonMobil (use XOM)',
        'NETFLIX': 'Netflix (use NFLX)',
        'ADOBE': 'Adobe (use ADBE)',
        'ORACLE': 'Oracle (use ORCL)',
        'IBM': 'IBM (use IBM)',
        'INTEL': 'Intel (use INTC)',
        'AMD': 'AMD (use AMD)',
        'VISA': 'Visa (use V)',
        'MASTERCARD': 'Mastercard (use MA)',
        'DISNEY': 'Disney (use DIS)',
        'COCA': 'Coca-Cola (use KO)',
        'WALMART': 'Walmart (use WMT)'
    }
    
    return jsonify({
        'supported_companies': symbol_mappings,
        'note': 'For Indian companies, you can also try adding .NS (NSE) suffix to the company name'
    })

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ------------------ PAPER TRADING ------------------
@app.route('/paper-trade')
def paper_trade():
    return render_template('paper_trade.html')

@app.route('/api/portfolio')
def get_portfolio():
    return jsonify(trading_engine.get_portfolio(1))

@app.route('/api/positions')
def get_positions():
    return jsonify(trading_engine.get_positions(1))

@app.route('/api/trades')
def get_trades():
    return jsonify(trading_engine.get_trade_history(1))

@app.route('/api/place_order', methods=['POST'])
def place_order():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    result = trading_engine.place_order(
        user_id=1,
        symbol=data.get('symbol'),
        side=data.get('side'),
        quantity=data.get('quantity'),
        price=data.get('price')
    )
    return jsonify(result)

# ------------------ SOCKET.IO ------------------
@socketio.on('connect')
def handle_connect():
    join_room('market_data')
    emit('market_update', stock_simulator.get_all_prices())

@socketio.on('disconnect')
def handle_disconnect():
    leave_room('market_data')

def broadcast_market_updates():
    while True:
        time.sleep(1)
        stock_simulator.update_prices()
        socketio.emit('market_update', stock_simulator.get_all_prices(), to='market_data')

# ------------------ NEWS ------------------
@app.route('/news')
def news_home():
    return render_template('news.html', symbol=None, news_data=None)

@app.route('/news/<symbol>')
def news_analysis(symbol):
    try:
        news_data = analyze_news(symbol.upper())
        return render_template('news.html', symbol=symbol.upper(), news_data=news_data)
    except Exception as e:
        logging.error(f"News error: {e}")
        return render_template('news.html', symbol=symbol.upper(), news_data=None, error=str(e))

@app.route('/api/news/<symbol>')
def api_news(symbol):
    try:
        return jsonify(analyze_news(symbol.upper()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ SETTINGS & ALERTS ------------------
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        if request.form.get('password'):
            user.password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user.alerts_enabled = bool(request.form.get('alerts'))
        if request.form.get('balance'):
            user.balance = float(request.form['balance'])
        if 'test_alert' in request.form and user.alerts_enabled:
            send_telegram_alert("Test Alert: Telegram is working!")
        db.session.commit()
        flash("Settings updated!", "success")
    return render_template('settings.html', user=user)

@app.route('/alerts', methods=['GET', 'POST'])
def alerts_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    if request.method == 'POST':
        symbol = request.form.get('symbol').upper()
        condition = request.form.get('condition')
        if symbol and condition:
            new_alert = Alert(user_id=user_id, symbol=symbol, condition=condition, status="Active")
            db.session.add(new_alert)
            db.session.commit()
            flash(f"Alert created for {symbol}: {condition}", "success")
    alerts = Alert.query.filter_by(user_id=user_id).all()
    return render_template('alerts.html', alerts=alerts)

@app.route('/check_alerts')
def run_alerts():
    check_alerts()
    flash("Alerts checked!", "info")
    return redirect(url_for('alerts_page'))

# ------------------ MAIN ------------------
def create_tables():
    with app.app_context():
        db.create_all()
        if not Portfolio.query.filter_by(user_id=1).first():
            portfolio = Portfolio(user_id=1, balance=100000.0, total_value=100000.0)
            db.session.add(portfolio)
            db.session.commit()

if __name__ == '__main__':
    create_tables()
    market_thread = threading.Thread(target=broadcast_market_updates, daemon=True)
    market_thread.start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
