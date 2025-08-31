from models import Portfolio, Position, Trade
from models import db

from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PaperTradingEngine:
    def __init__(self, database):
        self.db = database
        
    def get_portfolio(self, user_id):
        """Get user's portfolio information"""
        portfolio = Portfolio.query.filter_by(user_id=user_id).first()
        if not portfolio:
            # Create default portfolio
            portfolio = Portfolio(user_id=user_id, balance=100000.0, total_value=100000.0)
            self.db.session.add(portfolio)
            self.db.session.commit()
        
        # Update portfolio value with current positions
        self.update_portfolio_value(user_id)
        return portfolio.to_dict()
    
    def get_positions(self, user_id):
        """Get user's current positions"""
        positions = Position.query.filter_by(user_id=user_id).all()
        return [pos.to_dict() for pos in positions]
    
    def get_trade_history(self, user_id, limit=50):
        """Get user's trade history"""
        trades = Trade.query.filter_by(user_id=user_id)\
                           .order_by(Trade.executed_at.desc())\
                           .limit(limit).all()
        return [trade.to_dict() for trade in trades]
    
    def place_order(self, user_id, symbol, side, quantity, price=None):
        """Place a buy or sell order"""
        try:
            # Get current market price (simulated)
            from stock_simulator import StockSimulator
            simulator = StockSimulator()
            market_price = simulator.get_price(symbol)
            
            if not market_price:
                return {'success': False, 'error': 'Invalid symbol'}
            
            execution_price = price if price else market_price
            total_amount = execution_price * quantity
            
            # Get user's portfolio
            portfolio = Portfolio.query.filter_by(user_id=user_id).first()
            if not portfolio:
                return {'success': False, 'error': 'Portfolio not found'}
            
            if side.upper() == 'BUY':
                # Check if user has enough balance
                if portfolio.balance < total_amount:
                    return {'success': False, 'error': 'Insufficient funds'}
                
                # Update portfolio balance
                portfolio.balance -= total_amount
                
                # Update or create position
                position = Position.query.filter_by(user_id=user_id, symbol=symbol).first()
                if position:
                    # Calculate new average price
                    total_shares = position.quantity + quantity
                    total_cost = (position.avg_price * position.quantity) + total_amount
                    position.avg_price = total_cost / total_shares
                    position.quantity = total_shares
                else:
                    # Create new position
                    position = Position(
                        user_id=user_id,
                        symbol=symbol,
                        quantity=quantity,
                        avg_price=execution_price
                    )
                    self.db.session.add(position)
                
            elif side.upper() == 'SELL':
                # Check if user has enough shares
                position = Position.query.filter_by(user_id=user_id, symbol=symbol).first()
                if not position or position.quantity < quantity:
                    return {'success': False, 'error': 'Insufficient shares'}
                
                # Update portfolio balance
                portfolio.balance += total_amount
                
                # Update position
                position.quantity -= quantity
                if position.quantity == 0:
                    self.db.session.delete(position)
            
            else:
                return {'success': False, 'error': 'Invalid side'}
            
            # Create trade record
            trade = Trade(
                user_id=user_id,
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                price=execution_price,
                total_amount=total_amount,
                commission=0.0  # No commission for paper trading
            )
            self.db.session.add(trade)
            
            # Update portfolio timestamp
            portfolio.updated_at = datetime.utcnow()
            
            self.db.session.commit()
            
            logger.info(f"Order executed: {side} {quantity} {symbol} at ${execution_price}")
            
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
        """Update portfolio total value based on current positions"""
        try:
            portfolio = Portfolio.query.filter_by(user_id=user_id).first()
            if not portfolio:
                return
            
            positions = Position.query.filter_by(user_id=user_id).all()
            
            from stock_simulator import StockSimulator
            simulator = StockSimulator()
            
            total_position_value = 0.0
            
            for position in positions:
                current_price = simulator.get_price(position.symbol)
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
