import os
import re
import time
import math
import json
import random
import logging
import threading
import datetime
import secrets
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
from sqlalchemy.orm import DeclarativeBase
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import smtplib
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    MimeText = None
    MimeMultipart = None

# Local imports
from models import User, Portfolio, Trade, Position, Alert, db, Transaction, UserSettings, PasswordResetToken
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
app.secret_key = os.environ.get("SESSION_SECRET")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_recycle": 300, "pool_pre_ping": True}
db.init_app(app)

# Extensions
bcrypt = Bcrypt(app)

# Trading engines
trading_engine = PaperTradingEngine(db)
stock_simulator = StockSimulator()
predictor = StockPredictor()

# Global variable to hold the latest price updates from the background task
latest_prices = {}

# ------------------ HELPERS ------------------
def send_email(to_email, subject, body):
    with app.app_context():
        try:
            if not MimeText or not MimeMultipart:
                logging.warning("Email MIME modules not available")
                return False

            smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            smtp_username = os.getenv("SMTP_USERNAME")
            smtp_password = os.getenv("SMTP_PASSWORD")

            if not smtp_username or not smtp_password:
                logging.warning("Email credentials not configured")
                return False

            msg = MimeMultipart()
            msg['From'] = smtp_username
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MimeText(body, 'html'))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            logging.error(f"Email sending failed: {e}")
            return False

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
                alert.status = "Triggered"
                db.session.commit()
        except Exception as e:
            logging.error(f"Error checking alert: {e}")

def is_valid_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def requires_auth(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# ------------------ AUTH ROUTES ------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = request.form.get('remember_me', False)

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username

            if remember:
                session.permanent = True
                app.permanent_session_lifetime = timedelta(days=30)

            user.last_login = datetime.utcnow()
            db.session.commit()

            flash("Welcome back!", "success")
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return render_template('register.html')

        if not is_valid_email(email):
            flash("Please enter a valid email address.", "danger")
            return render_template('register.html')

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('register.html')

        is_valid, message = is_valid_password(password)
        if not is_valid:
            flash(message, "danger")
            return render_template('register.html')

        user = User(
            username=username,
            email=email,
            password=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()

        settings = UserSettings(user_id=user.id)
        db.session.add(settings)
        db.session.commit()

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()

        if user:
            token = secrets.token_urlsafe(32)
            reset_token = PasswordResetToken(
                user_id=user.id,
                token=token,
                expires_at=datetime.utcnow() + timedelta(hours=1)
            )
            db.session.add(reset_token)
            db.session.commit()

            reset_url = url_for('reset_password', token=token, _external=True)
            subject = "Password Reset - CashOnDay"
            body = f"""
            <html>
            <body>
                <h2>Password Reset Request</h2>
                <p>You requested a password reset for your CashOnDay account.</p>
                <p>Click the link below to reset your password:</p>
                <a href="{reset_url}" style="background-color: #0f4c75; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Reset Password</a>
                <p>This link will expire in 1 hour.</p>
                <p>If you didn't request this reset, please ignore this email.</p>
            </body>
            </html>
            """

            if send_email(email, subject, body):
                flash("Password reset instructions sent to your email.", "success")
            else:
                flash("Error sending email. Please try again later.", "danger")
        else:
            flash("If an account with that email exists, you'll receive reset instructions.", "info")

        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    reset_token = PasswordResetToken.query.filter_by(
        token=token,
        used=False
    ).first()

    if not reset_token or reset_token.expires_at < datetime.utcnow():
        flash("Invalid or expired reset token.", "danger")
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('reset_password.html', token=token)

        is_valid, message = is_valid_password(password)
        if not is_valid:
            flash(message, "danger")
            return render_template('reset_password.html', token=token)

        user = User.query.get(reset_token.user_id)
        user.password = generate_password_hash(password)
        reset_token.used = True
        db.session.commit()

        flash("Password reset successful. Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# ------------------ MAIN ROUTES ------------------
@app.route('/')
def home():
    return redirect(url_for('dashboard')) if 'user_id' in session else redirect(url_for('login'))

@app.route('/dashboard')
@requires_auth
def dashboard():
    user = User.query.get(session['user_id'])

    portfolio = trading_engine.get_portfolio(user.id)
    positions = trading_engine.get_positions(user.id)
    recent_trades = trading_engine.get_trade_history(user.id, limit=5)

    watchlist_symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]
    watchlist = []
    for sym in watchlist_symbols:
        try:
            data = yf.Ticker(sym).history(period="2d")
            if not data.empty:
                price = round(data['Close'].iloc[-1], 2)
                prev = round(data['Close'].iloc[-2], 2) if len(data) > 1 else price
                change = round(((price - prev) / prev) * 100, 2)
                watchlist.append({"symbol": sym, "price": price, "change": change})
        except:
            continue

    recent_activities = []

    from models import Prediction
    recent_predictions = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.created_at.desc()).limit(3).all()
    for pred in recent_predictions:
        recent_activities.append({
            'type': 'prediction',
            'description': f'Predicted {pred.stock_symbol}',
            'timestamp': pred.created_at,
            'icon': 'trending-up'
        })

    for trade in recent_trades:
        recent_activities.append({
            'type': 'trade',
            'description': f'{trade["side"]} {trade["quantity"]} {trade["symbol"]}',
            'timestamp': datetime.fromisoformat(trade['executed_at']),
            'icon': 'activity'
        })

    recent_activities.sort(key=lambda x: x['timestamp'], reverse=True)
    recent_activities = recent_activities[:10]

    return render_template('dashboard.html',
                           user=user,
                           portfolio=portfolio,
                           positions=positions,
                           watchlist=watchlist,
                           recent_trades=recent_trades,
                           recent_activities=recent_activities)

@app.route('/profile', methods=['GET', 'POST'])
@requires_auth
def profile():
    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        user.email = request.form['email']
        user.full_name = request.form.get('full_name', '')
        user.phone = request.form.get('phone', '')

        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')

        if current_password and new_password:
            if check_password_hash(user.password, current_password):
                is_valid, message = is_valid_password(new_password)
                if is_valid:
                    user.password = generate_password_hash(new_password)
                    flash("Password updated successfully.", "success")
                else:
                    flash(message, "danger")
            else:
                flash("Current password is incorrect.", "danger")

        db.session.commit()
        flash("Profile updated successfully.", "success")
        return redirect(url_for('profile'))

    transactions = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.timestamp.desc()).limit(50).all()
    return render_template('profile.html', user=user, transactions=transactions)

# ------------------ PREDICTION ROUTES ------------------
api_key = "698c83bbc29e4cfca8a617cb5783f809"
predictor = StockPredictor(twelve_api_key=api_key)

result = predictor.predict(symbol="AAPL")
print(result)

symbols = ["AAPL", "MSFT", "GOOGL", "RELIANCE.NS"]
for sym in symbols:
    res = predictor.predict(symbol=sym)
    print(sym, res)

predictor = StockPredictor(twelve_api_key=api_key)
@app.route('/predict', methods=['GET', 'POST'])
@requires_auth
def predict():
    return render_template('prediction.html')

@app.route('/api/stock-data/<symbol>')
@requires_auth
def get_stock_data(symbol):
    try:
        if not symbol or len(symbol) > 20:
            return jsonify({'error': 'Invalid stock symbol'}), 400

        symbol = symbol.upper().strip()
        symbol_variations = get_stock_symbol_variations(symbol)

        stock_data = None
        working_symbol = None

        for try_symbol in symbol_variations:
            try:
                stock = yf.Ticker(try_symbol)
                hist = stock.history(period="1y")
                if not hist.empty and len(hist) > 10:
                    stock_data = stock
                    working_symbol = try_symbol
                    break
            except Exception as e:
                logging.debug(f"Failed to fetch data for {try_symbol}: {str(e)}")
                continue

        if stock_data is None:
            return jsonify({'error': f'No data found for {symbol}. Please check the symbol or try the full exchange symbol (e.g., TCS.NS for Indian stocks)'}), 404

        symbol = working_symbol
        stock = stock_data
        hist = stock.history(period="1y")

        if hist.empty:
            return jsonify({'error': f'No data found for symbol {symbol}'}), 404

        info = stock.info

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

        current_price = round(float(hist['Close'].iloc[-1]), 2)
        prev_close = round(float(hist['Close'].iloc[-2]), 2)
        price_change = round(current_price - prev_close, 2)
        price_change_percent = round((price_change / prev_close) * 100, 2)

        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])

        prediction_result = predictor.predict(hist)

        if prediction_result:
            from models import Prediction
            prediction = Prediction(
                user_id=session['user_id'],
                stock_symbol=symbol,
                predicted_price=prediction_result['predicted_price'],
                confidence_score=prediction_result['confidence'],
                created_at=datetime.utcnow()
            )
            db.session.add(prediction)
            db.session.commit()

        response_data = {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'current_price': current_price,
            'price_change': price_change,
            'price_change_percent': price_change_percent,
            'chart_data': chart_data,
            'technical_indicators': {
                'sma_20': round(float(hist['SMA_20'].iloc[-1]), 2) if not np.isnan(hist['SMA_20'].iloc[-1]) else None,
                'sma_50': round(float(hist['SMA_50'].iloc[-1]), 2) if not np.isnan(hist['SMA_50'].iloc[-1]) else None,
                'rsi': round(float(hist['RSI'].iloc[-1]), 2) if not np.isnan(hist['RSI'].iloc[-1]) else None,
            },
            'prediction': prediction_result,
            'volume': int(hist['Volume'].iloc[-1]),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'dividend_yield': info.get('dividendYield')
        }

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error fetching stock data: {str(e)}")
        return jsonify({'error': 'An error occurred while fetching stock data'}), 500

def get_stock_symbol_variations(symbol):
    symbol = symbol.upper().strip()
    symbol_mappings = {
        'TCS': ['INFY', 'TCS.NS'],
        'TATA': ['TTM', 'TATAMOTORS.NS'],
        'RELIANCE': ['RELIANCE.NS'],
        'INFOSYS': ['INFY', 'INFY.NS'],
        'WIPRO': ['WIT', 'WIPRO.NS'],
        'HDFC': ['HDB', 'HDFCBANK.NS'],
        'ICICI': ['IBN', 'ICICIBANK.NS'],
        'MICROSOFT': ['MSFT'],
        'APPLE': ['AAPL'],
        'GOOGLE': ['GOOGL', 'GOOG'],
        'AMAZON': ['AMZN'],
        'TESLA': ['TSLA'],
        'META': ['META'],
        'NVIDIA': ['NVDA']
    }

    if symbol in symbol_mappings:
        return symbol_mappings[symbol]

    variations = [symbol]
    if '.' not in symbol:
        variations.extend([
            f"{symbol}.NS",
            f"{symbol}.BO",
            f"{symbol}.L",
            f"{symbol}.DE",
            f"{symbol}.PA"
        ])

    return variations

# ------------------ TRADING ROUTES ------------------
@app.route('/trading')
@requires_auth
def trading():
    user = User.query.get(session['user_id'])
    portfolio = trading_engine.get_portfolio(user.id)
    positions = trading_engine.get_positions(user.id)
    trades = trading_engine.get_trade_history(user.id, limit=20)

    return render_template('trading.html',
                           user=user,
                           portfolio=portfolio,
                           positions=positions,
                           trades=trades)

@app.route('/api/place-order', methods=['POST', 'GET'])
@requires_auth
def place_order():
    try:
        data = request.get_json()
        symbol = data['symbol'].upper()
        side = data['side'].upper()
        quantity = int(data['quantity'])
        order_type = data.get('order_type', 'market')
        price = float(data.get('price', 0)) if data.get('price') else None

        if quantity <= 0:
            return jsonify({'success': False, 'error': 'Invalid quantity'})

        if side not in ['BUY', 'SELL']:
            return jsonify({'success': False, 'error': 'Invalid order side'})

        result = trading_engine.place_order(session['user_id'], symbol, side, quantity, price)

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error placing order: {e}")
        return jsonify({'success': False, 'error': str(e)})

# ------------------ NEWS ROUTES ------------------
@app.route('/news')
@requires_auth
def news():
    return render_template('news.html')

@app.route('/api/news/<symbol>')
@requires_auth
def get_news(symbol):
    try:
        result = analyze_news(symbol)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error fetching news: {e}")
        return jsonify({'error': str(e)}), 500

# ------------------ SETTINGS ROUTES ------------------
@app.route('/settings', methods=['GET', 'POST'])
@requires_auth
def settings():
    user = User.query.get(session['user_id'])
    user_settings = UserSettings.query.filter_by(user_id=user.id).first()

    if not user_settings:
        user_settings = UserSettings(user_id=user.id)
        db.session.add(user_settings)
        db.session.commit()

    if request.method == 'POST':
        user_settings.email_notifications = 'email_notifications' in request.form
        user_settings.price_alerts = 'price_alerts' in request.form
        user_settings.news_alerts = 'news_alerts' in request.form
        user_settings.theme = request.form.get('theme', 'light')

        db.session.commit()
        flash("Settings updated successfully.", "success")
        return redirect(url_for('settings'))

    return render_template('settings.html', user=user, settings=user_settings)

@app.route('/alerts')
@requires_auth
def alerts():
    user_id = session['user_id']
    user_alerts = Alert.query.filter_by(user_id=user_id).order_by(Alert.created_at.desc()).all()
    user = User.query.get(user_id)
    return render_template('alerts.html', alerts=user_alerts, user=user)

@app.route('/api/create-alert', methods=['POST'])
@requires_auth
def create_alert():
    try:
        data = request.get_json()
        symbol = data['symbol'].upper()
        condition = data['condition']

        if not re.match(r"Price\s*([<>]=?|==)\s*(\d+\.?\d*)", condition):
            return jsonify({'success': False, 'error': 'Invalid condition format'})

        alert = Alert(
            user_id=session['user_id'],
            symbol=symbol,
            condition=condition,
            status="Active",
            created_at=datetime.utcnow()
        )
        db.session.add(alert)
        db.session.commit()

        return jsonify({'success': True, 'alert_id': alert.id})

    except Exception as e:
        logging.error(f"Error creating alert: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete-alert/<int:alert_id>', methods=['DELETE'])
@requires_auth
def delete_alert(alert_id):
    try:
        alert = Alert.query.filter_by(id=alert_id, user_id=session['user_id']).first()
        if alert:
            db.session.delete(alert)
            db.session.commit()
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Alert not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ------------------ BACKGROUND TASKS AND DATA API ------------------
def background_tasks():
    global latest_prices
    while True:
        with app.app_context():
            try:
                stock_simulator.update_prices()
                check_alerts()
                latest_prices = stock_simulator.get_all_prices()
            except Exception as e:
                logging.error(f"Background task error: {e}")
        time.sleep(30)

@app.route("/api/realtime-prices")
def realtime_prices():
    symbols = ["AAPL", "GOOGL", "AMZN", "CRM", "ADBE"]
    prices = {}

    for symbol in symbols:
        ticker = yf.Ticker(symbol)

        # Try intraday (1-minute) data first
        data = ticker.history(period="1d", interval="1m")

        if not data.empty:
            last_row = data.iloc[-1]
            close_price = last_row["Close"]
            open_price = last_row["Open"]
        else:
            close_price, open_price = None, None

        # Fallback: if invalid, use 1-day data (yesterday's close)
        if close_price is None or math.isnan(close_price):
            day_data = ticker.history(period="2d", interval="1d")
            if not day_data.empty:
                close_price = day_data["Close"].iloc[-1]
                open_price = day_data["Open"].iloc[-1]

        # Final safety: replace with None if still NaN
        price = float(close_price) if close_price is not None and not math.isnan(close_price) else None
        change = (
            float(close_price - open_price)
            if (close_price is not None and open_price is not None and not math.isnan(close_price) and not math.isnan(open_price))
            else None
        )

        prices[symbol] = {"price": price, "change": change}

    return jsonify(prices)
# Start background thread
if __name__ == "__main__":
    background_thread = threading.Thread(target=background_tasks)
    background_thread.daemon = True
    background_thread.start()

    with app.app_context():
        db.create_all()

    # Render will set PORT env automatically
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
