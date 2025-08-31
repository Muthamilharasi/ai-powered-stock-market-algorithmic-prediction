# Overview

CashOnDay is an AI-powered stock prediction and paper trading web application built with Flask. The platform combines machine learning-based stock price prediction with virtual trading capabilities, allowing users to practice trading strategies using simulated money. The application features real-time market simulation, news sentiment analysis, price alerts, and Telegram integration for notifications.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Flask with Jinja2 templating engine
- **UI Components**: Bootstrap 5 for responsive design with custom CSS styling
- **Real-time Communication**: Socket.IO for WebSocket connections enabling live price updates and notifications
- **Charting**: Chart.js and Plotly for interactive financial charts and data visualization
- **Icons**: Feather Icons for consistent iconography
- **Theme System**: Light/dark theme support with CSS custom properties

## Backend Architecture
- **Web Framework**: Flask with modular design pattern
- **Authentication**: Session-based authentication with bcrypt password hashing
- **Real-time Features**: Flask-SocketIO for WebSocket support
- **Stock Data Simulation**: Custom StockSimulator class that generates realistic price movements using geometric Brownian motion
- **Trading Engine**: PaperTradingEngine for virtual trading operations
- **AI Prediction Engine**: StockPredictor using RandomForest and technical indicators
- **News Analysis**: Sentiment analysis using HuggingFace Transformers (DistilBERT)

## Data Storage
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Models**: Comprehensive user management, portfolio tracking, trading history, alerts, and settings
- **Session Management**: Flask sessions for user state persistence
- **Connection Pooling**: SQLAlchemy with pool recycling and pre-ping for reliability

## Core Features
- **AI Stock Prediction**: Machine learning models with technical indicators (SMA, RSI, MACD, volatility measures)
- **Paper Trading**: Virtual trading platform with real-time portfolio tracking
- **Price Alerts**: Customizable price monitoring with notification system
- **News Sentiment Analysis**: AI-powered news analysis for market insights
- **User Management**: Complete authentication system with password reset functionality

## Security & Authentication
- **Password Security**: Bcrypt hashing with salt
- **Session Management**: Secure session handling with configurable secrets
- **Input Validation**: Form validation and CSRF protection
- **Password Reset**: Token-based password recovery system

# External Dependencies

## Third-party Services
- **Telegram Bot API**: For delivering trading alerts and notifications to users
- **Yahoo Finance (yfinance)**: Stock data retrieval (with fallback to simulated data)
- **News APIs**: External news sources for sentiment analysis (with demo data fallback)

## Machine Learning & AI
- **Scikit-learn**: RandomForest and LinearRegression models for price prediction
- **HuggingFace Transformers**: DistilBERT model for news sentiment analysis
- **NumPy & Pandas**: Data manipulation and numerical computations

## Infrastructure
- **PostgreSQL**: Primary database for user data and trading records
- **SMTP Email Service**: Password reset and notification emails
- **WebSocket Support**: Real-time communication for live updates

## Key Libraries
- **Flask Ecosystem**: Flask-SQLAlchemy, Flask-SocketIO, Flask-Bcrypt
- **Data Processing**: pandas, numpy for financial calculations
- **Visualization**: plotly, chart.js for interactive charts
- **Security**: werkzeug for password utilities and proxy handling

The application is designed with graceful degradation - it can operate with demo data when external APIs are unavailable, ensuring consistent functionality regardless of external service availability.