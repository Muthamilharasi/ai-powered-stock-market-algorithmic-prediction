from models import Portfolio, Position, Trade, db
from datetime import datetime
import logging
import yfinance as yf

logger = logging.getLogger(__name__)

class PaperTradingEngine:
    def __init__(self, database):
        self.db = database

    def get_live_price(self, symbol):
        """Fetch live stock price with fallback handling"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d", interval="1m")  # intraday
            if hist.empty:
                hist = stock.history(period="5d", interval="1d")  # fallback

            if not hist.empty:
                price = hist['Close'].iloc[-1]
                return float(price)

            logger.warning(f"No price data found for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def get_portfolio(self, user_id):
        """Get or create user portfolio"""
        portfolio = Portfolio.query.filter_by(user_id=user_id).first()
        if not portfolio:
            portfolio = Portfolio(user_id=user_id, balance=100000.0, total_value=100000.0)
            self.db.session.add(portfolio)
            self.db.session.commit()

        self.update_portfolio_value(user_id)
        return portfolio.to_dict()

    def get_positions(self, user_id):
        """Return all positions"""
        positions = Position.query.filter_by(user_id=user_id).all()
        return [pos.to_dict() for pos in positions]

    def get_trade_history(self, user_id, limit=50):
        """Return recent trade history"""
        trades = Trade.query.filter_by(user_id=user_id) \
                            .order_by(Trade.executed_at.desc()) \
                            .limit(limit).all()
        return [trade.to_dict() for trade in trades]

    def place_order(self, user_id, symbol, side, quantity, price=None):
        """Execute Buy/Sell order"""
        try:
            market_price = self.get_live_price(symbol)
            if not market_price:
                return {'success': False, 'error': 'Could not fetch live price'}

            execution_price = price if price else market_price
            total_amount = execution_price * quantity

            portfolio = Portfolio.query.filter_by(user_id=user_id).first()
            if not portfolio:
                return {'success': False, 'error': 'Portfolio not found'}

            if side.upper() == 'BUY':
                if portfolio.balance < total_amount:
                    return {'success': False, 'error': 'Insufficient funds'}

                portfolio.balance -= total_amount

                position = Position.query.filter_by(user_id=user_id, symbol=symbol).first()
                if position:
                    total_shares = position.quantity + quantity
                    total_cost = (position.avg_price * position.quantity) + total_amount
                    position.avg_price = total_cost / total_shares
                    position.quantity = total_shares
                else:
                    position = Position(
                        user_id=user_id,
                        symbol=symbol,
                        quantity=quantity,
                        avg_price=execution_price
                    )
                    self.db.session.add(position)

            elif side.upper() == 'SELL':
                position = Position.query.filter_by(user_id=user_id, symbol=symbol).first()
                if not position or position.quantity < quantity:
                    return {'success': False, 'error': 'Insufficient shares'}

                portfolio.balance += total_amount
                position.quantity -= quantity
                if position.quantity == 0:
                    self.db.session.delete(position)

            else:
                return {'success': False, 'error': 'Invalid side'}

            trade = Trade(
                user_id=user_id,
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                price=execution_price,
                total_amount=total_amount,
                commission=0.0
            )
            self.db.session.add(trade)

            portfolio.updated_at = datetime.utcnow()
            self.db.session.commit()

            return {
                'success': True,
                'trade_id': trade.id,
                'executed_price': execution_price,
                'total_amount': total_amount
            }

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            self.db.session.rollback()
            return {'success': False, 'error': str(e)}

    def update_portfolio_value(self, user_id):
        """Recalculate portfolio total value"""
        try:
            portfolio = Portfolio.query.filter_by(user_id=user_id).first()
            if not portfolio:
                return

            positions = Position.query.filter_by(user_id=user_id).all()
            total_position_value = 0.0

            for position in positions:
                current_price = self.get_live_price(position.symbol)
                if current_price:
                    position.current_price = current_price
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = position.market_value - (position.quantity * position.avg_price)
                    total_position_value += position.market_value

            portfolio.total_value = portfolio.balance + total_position_value
            self.db.session.commit()

        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
            self.db.session.rollback()
