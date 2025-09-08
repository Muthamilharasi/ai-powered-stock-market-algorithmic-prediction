import os
import re
import time
import logging
from datetime import datetime
import warnings

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logging.warning("requests not available - using demo data")
    REQUESTS_AVAILABLE = False

warnings.filterwarnings("ignore")

try:
    from transformers import pipeline
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        return_all_scores=True
    )
    logging.info("Sentiment analysis model loaded successfully")
except Exception as e:
    logging.warning("Transformers not available or error loading model, using fallback sentiment analysis")
    sentiment_analyzer = None

def get_company_name(symbol):
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
    if not text:
        return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 'Low'}
    text_lower = text.lower()
    positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'rise', 'gain', 'profit', 'growth',
                      'strong', 'buy', 'bullish', 'surge', 'boost', 'increase', 'beat', 'exceed', 'outperform', 'success']
    negative_words = ['bad', 'terrible', 'negative', 'down', 'fall', 'loss', 'decline', 'weak', 'sell',
                      'bearish', 'drop', 'crash', 'decrease', 'miss', 'underperform', 'fail', 'concern', 'worry']
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
    return {'label': label, 'score': round(sentiment_score, 3), 'confidence': confidence}

def analyze_sentiment(text):
    if not text:
        return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 'Low'}
    if sentiment_analyzer:
        try:
            results = sentiment_analyzer(text[:512])
            best_result = max(results[0], key=lambda x: x['score'])
            label_map = {'POSITIVE': 'POSITIVE', 'NEGATIVE': 'NEGATIVE'}
            sentiment_label = label_map.get(best_result['label'], 'NEUTRAL')
            confidence_score = best_result['score']
            fallback = simple_sentiment_analysis(text)
            if sentiment_label == fallback['label']:
                combined_confidence = max(confidence_score, fallback['score'])
            else:
                combined_confidence = confidence_score * 0.7 + fallback['score'] * 0.3
            confidence_level = 'High' if combined_confidence > 0.8 else 'Medium' if combined_confidence > 0.6 else 'Low'
            return {'label': sentiment_label, 'score': round(combined_confidence,3), 'confidence': confidence_level}
        except Exception as e:
            logging.error(f"Error in HuggingFace sentiment analysis: {e}")
    return simple_sentiment_analysis(text)

def text_contains_term(text, terms):
    text_lower = text.lower()
    for term in terms:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        if re.search(pattern, text_lower):
            return True
    return False

def popup_error(message):
    # Connect this function to your UI logic to show error popup dialogs to users
    print(f"ERROR POPUP: {message}")  # placeholder print statement

def fetch_relevant_news(symbol, limit=10, retries=3):
    if not REQUESTS_AVAILABLE:
        logging.info("Requests not available, cannot fetch real news")
        return []

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        popup_error("API key for NewsAPI is missing. Please set NEWS_API_KEY environment variable.")
        return []

    company_name = get_company_name(symbol)
    query = f'"{symbol}" OR "{company_name}"'

    url = "https://newsapi.org/v2/everything"
    params = {
        'q': query,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': limit,
        'apiKey': api_key
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data['status'] == 'ok' and data['totalResults'] > 0:
                return data['articles']
            elif data['totalResults'] == 0:
                logging.warning(f"No news found for {symbol}")
                return []
            else:
                logging.warning(f"NewsAPI returned status={data['status']} with no articles.")
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    popup_error(f"Failed to fetch news for {symbol} after {retries} attempts.")
    return []

def analyze_news(symbol):
    logging.info(f"Fetching news for {symbol}")
    articles = fetch_relevant_news(symbol)
    if articles is None or len(articles) == 0:
        popup_error(f"No stock market relevant news found for '{symbol}'. Please check the symbol or company name.")
        return None

    company_name = get_company_name(symbol)
    search_terms = [symbol.lower(), company_name.lower()]

    relevant_articles = []
    sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}

    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')

        if text_contains_term(title, search_terms) or text_contains_term(description, search_terms):
            text_to_analyze = title + " " + description
            sentiment = analyze_sentiment(text_to_analyze)

            try:
                pub_date = datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00'))
                published_at = pub_date.strftime('%Y-%m-%d %H:%M UTC')
            except Exception:
                published_at = 'Unknown'

            relevant_articles.append({
                'title': title,
                'description': description,
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'publishedAt': published_at,
                'sentiment': sentiment
            })
            sentiment_counts[sentiment['label']] += 1

    if not relevant_articles:
        popup_error(f"No relevant stock news found after filtering for '{symbol}'.")
        return None

    total = len(relevant_articles)
    sentiment_distribution = {
        'positive': round((sentiment_counts['POSITIVE'] / total) * 100, 1),
        'negative': round((sentiment_counts['NEGATIVE'] / total) * 100, 1),
        'neutral': round((sentiment_counts['NEUTRAL'] / total) * 100, 1)
    }
    overall_score = (sentiment_counts['POSITIVE'] - sentiment_counts['NEGATIVE']) / total
    overall_sentiment = 'POSITIVE' if overall_score > 0.1 else 'NEGATIVE' if overall_score < -0.1 else 'NEUTRAL'

    return {
        'symbol': symbol,
        'articles': relevant_articles,
        'sentiment_distribution': sentiment_distribution,
        'sentiment_counts': sentiment_counts,
        'overall_sentiment': overall_sentiment,
        'overall_score': round(overall_score, 3),
        'total_articles': total,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    }

# Example usage:
if __name__ == "__main__":
    stock_symbol = input("Enter stock symbol or company name: ").strip()
    result = analyze_news(stock_symbol)
    if result:
        print(f"Fetched {result['total_articles']} relevant news articles for {result['symbol']}.")
        for art in result['articles']:
            print(f"{art['publishedAt']} | {art['source']} | {art['title']} | Sentiment: {art['sentiment']['label']} ({art['sentiment']['score']})")
            print(f"URL: {art['url']}\n")
