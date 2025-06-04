# utils.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from transformers import pipeline
from newspaper import Article


def download_nltk_data():
    """Download required NLTK data only if missing."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')


def preprocess_text(text):
    """Basic text cleaning and lemmatization."""
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text)  # URLs
    text = re.sub(r'[^\w\s]', '', text)  # punctuation
    text = re.sub(r'\d+', '', text)  # numbers
    text = ' '.join(text.split())  # extra spaces

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def load_and_preprocess_data(fake_file, true_file):
    """Load CSVs, label, and preprocess."""
    fake_df = pd.read_csv(fake_file)
    true_df = pd.read_csv(true_file)

    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    # Use both title + text (for better generalization)
    df['combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['processed_text'] = df['combined'].apply(preprocess_text)

    return df['processed_text'], df['label']


summarizer = None

def summarize_text(text):
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    if len(text.split()) > 1000:
        text = ' '.join(text.split()[:1000])
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']


def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text.strip()) > 100:
            return article.text.strip()
    except:
        pass  # Fall back to manual extraction

    # Fallback using BeautifulSoup
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs)
        return text.strip() if len(text.strip()) > 100 else None
    except:
        return None


def analyze_text_characteristics(text):
    characteristics = []
    clickbait_words = ['shocking', 'unbelievable', 'you won\'t believe', 'breaking', 'miracle']
    emotional_words = ['horrifying', 'terrifying', 'heartbreaking', 'devastating']
    satire_signs = ['the onion', 'babylon bee', 'satire']
    unreal_claims = ['earth is flat', 'aliens in white house', 'cure in 24 hours','alien']

    lower_text = text.lower()
    if any(word in lower_text for word in clickbait_words):
        characteristics.append("Possible clickbait")
    if any(word in lower_text for word in emotional_words):
        characteristics.append("Emotional language")
    if any(word in lower_text for word in satire_signs):
        characteristics.append("May be satire")
    if any(word in lower_text for word in unreal_claims):
        characteristics.append("Unrealistic claim")
    if any(char.isdigit() for char in text):
        characteristics.append("Statistical references present")
    return characteristics


def analyze_source_credibility(url):
    try:
        domain = urlparse(url).netloc.lower()
        credible_sources = ['bbc.co.uk', 'cnn.com', 'reuters.com', 'thehindu.com', 'ndtv.com', 'who.int', 'un.org']
        if any(cred in domain for cred in credible_sources):
            return f"Source appears credible ({domain})"
        else:
            return f"Source is not in verified list ({domain})"
    except:
        return "Source unknown or invalid URL."