import os
import logging
from telegram import Bot
from telegram.error import TelegramError

class TelegramBotHandler:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_TOKEN")
        self.bot = None
        
        if self.bot_token:
            try:
                self.bot = Bot(token=self.bot_token)
                logging.info("Telegram bot initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Telegram bot: {e}")
        else:
            logging.warning("TELEGRAM_TOKEN not found in environment variables")
    
    def send_alert(self, chat_id, message):
        """Send alert message to specified chat"""
        if not self.bot or not chat_id:
            logging.warning("Telegram bot or chat_id not available")
            return False
            
        try:
            self.bot.send_message(chat_id=chat_id, text=message)
            logging.info(f"Alert sent to {chat_id}: {message}")
            return True
        except TelegramError as e:
            logging.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_price_alert(self, chat_id, symbol, current_price, condition):
        """Send formatted price alert"""
        message = f"🚨 Price Alert Triggered!\n\n"
        message += f"Symbol: {symbol}\n"
        message += f"Current Price: ${current_price:.2f}\n"
        message += f"Condition: {condition}\n"
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        return self.send_alert(chat_id, message)
    
    def send_trade_notification(self, chat_id, trade_info):
        """Send trade execution notification"""
        message = f"✅ Trade Executed!\n\n"
        message += f"Action: {trade_info['side']}\n"
        message += f"Symbol: {trade_info['symbol']}\n"
        message += f"Quantity: {trade_info['quantity']}\n"
        message += f"Price: ${trade_info['price']:.2f}\n"
        message += f"Total: ${trade_info['total_amount']:.2f}\n"
        message += f"Time: {trade_info['executed_at']}"
        
        return self.send_alert(chat_id, message)
    
    def verify_chat_id(self, chat_id):
        """Verify if chat_id is valid"""
        if not self.bot:
            return False
            
        try:
            self.bot.send_message(chat_id=chat_id, text="✅ Chat ID verified successfully!")
            return True
        except TelegramError:
            return False
