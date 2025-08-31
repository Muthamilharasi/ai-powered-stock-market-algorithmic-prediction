import os
import logging
from datetime import datetime
import warnings

# Optional imports - handle missing packages gracefully
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logging.warning("requests not available - using demo data")
    REQUESTS_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    logging.warning("yfinance not available - using demo data")
    YFINANCE_AVAILABLE = False

# Suppress warnings from transformers
warnings.filterwarnings("ignore")

# Initialize sentiment analysis pipeline
sentiment_analyzer = None
try:
    from transformers import pipeline
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        return_all_scores=True
    )
    logging.info("Sentiment analysis model loaded successfully")
except ImportError:
    logging.warning("Transformers not available, using fallback sentiment analysis")
    sentiment_analyzer = None
except Exception as e:
    logging.error(f"Error loading sentiment model: {e}")
    sentiment_analyzer = None

def get_demo_news(symbol, limit=10):
    """Generate demo news data when APIs are not available"""
    demo_articles = [
        {
            'title': f'{symbol} Reports Strong Q4 Earnings, Beats Expectations',
            'description': f'{get_company_name(symbol)} announced impressive quarterly results with revenue growth exceeding analyst predictions.',
            'url': 'https://example.com/news1',
            'source': {'name': 'Demo Financial News'},
            'publishedAt': datetime.now().isoformat() + 'Z'
        },
        {
            'title': f'Analysts Upgrade {symbol} Stock Rating to Buy',
            'description': f'Major investment firms raise price targets for {get_company_name(symbol)} citing strong fundamentals.',
            'url': 'https://example.com/news2', 
            'source': {'name': 'Demo Market Watch'},
            'publishedAt': datetime.now().isoformat() + 'Z'
        },
        {
            'title': f'{symbol} Faces Regulatory Challenges in New Markets',
            'description': f'{get_company_name(symbol)} encounters potential headwinds from regulatory scrutiny in emerging markets.',
            'url': 'https://example.com/news3',
            'source': {'name': 'Demo Business Weekly'},
            'publishedAt': datetime.now().isoformat() + 'Z'
        },
        {
            'title': f'Innovation Drive: {symbol} Launches New Product Line',
            'description': f'{get_company_name(symbol)} unveils cutting-edge technology solutions to expand market reach.',
            'url': 'https://example.com/news4',
            'source': {'name': 'Demo Tech Today'},
            'publishedAt': datetime.now().isoformat() + 'Z'
        },
        {
            'title': f'{symbol} Stock Volatile Amid Market Uncertainty',
            'description': f'Shares of {get_company_name(symbol)} experience fluctuations due to broader market conditions.',
            'url': 'https://example.com/news5',
            'source': {'name': 'Demo Markets Daily'},
            'publishedAt': datetime.now().isoformat() + 'Z'
        }
    ]
    return demo_articles[:limit]

def fetch_news_newsapi(symbol, limit=10):
    """Fetch news using NewsAPI"""
    if not REQUESTS_AVAILABLE:
        logging.info("Using demo news data (requests not available)")
        return get_demo_news(symbol, limit)
        
    api_key = os.getenv("NEWS_API_KEY")
    
    if not api_key:
        logging.warning("NEWS_API_KEY not found in environment variables, using demo data")
        return get_demo_news(symbol, limit)
    
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': f'"{symbol}" OR "{get_company_name(symbol)}"',
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': limit,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'ok' and data['articles']:
            return data['articles']
        else:
            logging.warning(f"No articles found for {symbol}")
            return []
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news from NewsAPI: {e}")
        return None

def fetch_news_yahoo(symbol, limit=10):
    """Fetch news using Yahoo Finance (fallback method)"""
    if not YFINANCE_AVAILABLE:
        logging.info("Using demo news data (yfinance not available)")
        return get_demo_news(symbol, limit)
        
    try:
        
        ticker = yf.Ticker(symbol)
        news = ticker.news
        
        if not news:
            return []
        
        # Transform Yahoo Finance news format to match NewsAPI format
        articles = []
        for item in news[:limit]:
            articles.append({
                'title': item.get('title', ''),
                'description': item.get('summary', ''),
                'url': item.get('link', ''),
                'source': {'name': item.get('publisher', 'Yahoo Finance')},
                'publishedAt': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat() + 'Z'
            })
        
        return articles
        
    except ImportError:
        logging.error("yfinance not available")
        return None
    except Exception as e:
        logging.error(f"Error fetching news from Yahoo Finance: {e}")
        return None

def get_company_name(symbol):
    """Get company name for better search results"""
    company_names = {
        'AAPL': 'Apple Inc',
        'TSLA': 'Tesla Inc',
        'GOOGL': 'Google Alphabet',
        'MSFT': 'Microsoft Corporation',
        'AMZN': 'Amazon.com Inc',
        'NVDA': 'NVIDIA Corporation',
        'META': 'Meta Platforms Inc',
        'NFLX': 'Netflix Inc',
        'AMD': 'Advanced Micro Devices',
        'INTC': 'Intel Corporation'
    }
    return company_names.get(symbol.upper(), symbol)

def simple_sentiment_analysis(text):
    """Simple fallback sentiment analysis using keyword matching"""
    if not text:
        return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 'Low'}
    
    text_lower = text.lower()
    
    # Positive keywords
    positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'rise', 'gain', 'profit', 'growth', 'strong', 'buy', 'bullish', 'surge', 'boost', 'increase', 'beat', 'exceed', 'outperform', 'success']
    
    # Negative keywords
    negative_words = ['bad', 'terrible', 'negative', 'down', 'fall', 'loss', 'decline', 'weak', 'sell', 'bearish', 'drop', 'crash', 'decrease', 'miss', 'underperform', 'fail', 'concern', 'worry']
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total_sentiment_words = positive_count + negative_count
    
    if total_sentiment_words == 0:
        return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 'Low'}
    
    sentiment_score = positive_count / total_sentiment_words
    
    if sentiment_score > 0.6:
        label = 'POSITIVE'
        confidence = 'Medium' if sentiment_score > 0.8 else 'Low'
    elif sentiment_score < 0.4:
        label = 'NEGATIVE'
        confidence = 'Medium' if sentiment_score < 0.2 else 'Low'
    else:
        label = 'NEUTRAL'
        confidence = 'Low'
    
    return {
        'label': label,
        'score': round(sentiment_score, 3),
        'confidence': confidence
    }

def analyze_sentiment(text):
    """Analyze sentiment of a text using HuggingFace model or fallback"""
    if not text:
        return {
            'label': 'NEUTRAL',
            'score': 0.5,
            'confidence': 'Low'
        }
    
    # Try HuggingFace model first
    if sentiment_analyzer:
        try:
            # Get sentiment prediction
            results = sentiment_analyzer(text[:512])  # Limit text length
            
            # Extract the best prediction
            best_result = max(results[0], key=lambda x: x['score'])
            
            # Map labels and determine confidence
            label_map = {'POSITIVE': 'POSITIVE', 'NEGATIVE': 'NEGATIVE'}
            sentiment_label = label_map.get(best_result['label'], 'NEUTRAL')
            
            confidence_level = 'High' if best_result['score'] > 0.8 else 'Medium' if best_result['score'] > 0.6 else 'Low'
            
            return {
                'label': sentiment_label,
                'score': round(best_result['score'], 3),
                'confidence': confidence_level
            }
            
        except Exception as e:
            logging.error(f"Error in HuggingFace sentiment analysis: {e}")
    
    # Fallback to simple sentiment analysis
    return simple_sentiment_analysis(text)

def analyze_news(symbol):
    """Main function to fetch news and analyze sentiment"""
    logging.info(f"Starting news analysis for {symbol}")
    
    # Try NewsAPI first, then Yahoo Finance as fallback
    articles = fetch_news_newsapi(symbol)
    
    if articles is None:
        logging.info("NewsAPI failed, trying Yahoo Finance...")
        articles = fetch_news_yahoo(symbol)
    
    if articles is None:
        return {"error": "Failed to fetch news from any source. Please check your API keys and internet connection."}
    
    if not articles:
        return {"error": f"No news articles found for symbol {symbol}. Please try a different stock symbol."}
    
    # Analyze sentiment for each article
    analyzed_articles = []
    sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    
    for article in articles:
        # Combine title and description for better sentiment analysis
        text_to_analyze = f"{article.get('title', '')} {article.get('description', '')}"
        sentiment = analyze_sentiment(text_to_analyze)
        
        # Parse published date
        try:
            pub_date = datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00'))
            formatted_date = pub_date.strftime('%Y-%m-%d %H:%M UTC')
        except:
            formatted_date = 'Unknown'
        
        analyzed_article = {
            'title': article.get('title', 'No Title'),
            'description': article.get('description', ''),
            'url': article.get('url', ''),
            'source': article.get('source', {}).get('name', 'Unknown'),
            'publishedAt': formatted_date,
            'sentiment': sentiment
        }
        
        analyzed_articles.append(analyzed_article)
        sentiment_counts[sentiment['label']] += 1
    
    # Calculate sentiment percentages
    total_articles = len(analyzed_articles)
    sentiment_distribution = {
        'positive': round((sentiment_counts['POSITIVE'] / total_articles) * 100, 1),
        'negative': round((sentiment_counts['NEGATIVE'] / total_articles) * 100, 1),
        'neutral': round((sentiment_counts['NEUTRAL'] / total_articles) * 100, 1)
    }
    
    # Calculate overall sentiment score
    overall_score = (sentiment_counts['POSITIVE'] - sentiment_counts['NEGATIVE']) / total_articles
    overall_sentiment = 'POSITIVE' if overall_score > 0.1 else 'NEGATIVE' if overall_score < -0.1 else 'NEUTRAL'
    
    result = {
        'symbol': symbol,
        'articles': analyzed_articles,
        'sentiment_distribution': sentiment_distribution,
        'sentiment_counts': sentiment_counts,
        'overall_sentiment': overall_sentiment,
        'overall_score': round(overall_score, 3),
        'total_articles': total_articles,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    }
    
    logging.info(f"Successfully analyzed {total_articles} articles for {symbol}")
    return result
