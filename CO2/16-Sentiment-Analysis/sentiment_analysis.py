# ============================================
# SENTIMENT ANALYSIS USING TEXTBLOB AND VADER
# Google Colab - Complete Program in One Cell
# ============================================

# Install required libraries
!pip install -q textblob nltk

# Import libraries
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon', quiet=True)

# Create VADER analyzer
sia = SentimentIntensityAnalyzer()

# --------------------------------------------
# Sample texts
# --------------------------------------------
texts = [
    "I absolutely love this movie! It was amazing and wonderful.",
    "This product is terrible and I hate it.",
    "The movie was okay, nothing special.",
    "I am very happy with the excellent service.",
    "The service was extremely bad and disappointing.",
    "I went to the store today."
]

# --------------------------------------------
# Function for TextBlob sentiment
# --------------------------------------------
def textblob_sentiment(text):
    blob = TextBlob(text)

    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return polarity, subjectivity, sentiment


# --------------------------------------------
# Function for VADER sentiment
# --------------------------------------------
def vader_sentiment(text):
    scores = sia.polarity_scores(text)

    compound = scores['compound']

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return scores, sentiment


# --------------------------------------------
# Perform Sentiment Analysis
# --------------------------------------------
print("=" * 80)
print("SENTIMENT ANALYSIS USING TEXTBLOB AND VADER")
print("=" * 80)

for i, text in enumerate(texts, 1):

    # TextBlob
    tb_polarity, tb_subjectivity, tb_sentiment = textblob_sentiment(text)

    # VADER
    vader_scores, vader_result = vader_sentiment(text)

    print(f"\nText {i}: {text}")

    print("\n--- TextBlob ---")
    print(f"Polarity    : {tb_polarity:.2f}")
    print(f"Subjectivity: {tb_subjectivity:.2f}")
    print(f"Sentiment   : {tb_sentiment}")

    print("\n--- VADER ---")
    print(f"Positive Score: {vader_scores['pos']:.2f}")
    print(f"Negative Score: {vader_scores['neg']:.2f}")
    print(f"Neutral Score : {vader_scores['neu']:.2f}")
    print(f"Compound Score: {vader_scores['compound']:.2f}")
    print(f"Sentiment     : {vader_result}")

    print("-" * 80)


# --------------------------------------------
# Test your own sentence
# --------------------------------------------
print("\n\nCUSTOM TEXT SENTIMENT ANALYSIS")
print("=" * 80)

custom_text = input("Enter a sentence: ")

# TextBlob
tb_polarity, tb_subjectivity, tb_sentiment = textblob_sentiment(custom_text)

# VADER
vader_scores, vader_result = vader_sentiment(custom_text)

print("\nText:", custom_text)

print("\nTextBlob Result:")
print("Polarity    :", round(tb_polarity, 2))
print("Subjectivity:", round(tb_subjectivity, 2))
print("Sentiment   :", tb_sentiment)

print("\nVADER Result:")
print("Positive:", round(vader_scores['pos'], 2))
print("Negative:", round(vader_scores['neg'], 2))
print("Neutral :", round(vader_scores['neu'], 2))
print("Compound:", round(vader_scores['compound'], 2))
print("Sentiment:", vader_result)


# --------------------------------------------
# Simple comparison
# --------------------------------------------
print("\n\nCOMPARISON")
print("=" * 80)

print("""
TextBlob:
- Uses polarity and subjectivity.
- Polarity ranges from -1 to +1.
- Subjectivity ranges from 0 to 1.
- Good for general/simple sentiment analysis.

VADER:
- Uses Positive, Negative, Neutral and Compound scores.
- Compound score ranges from -1 to +1.
- Particularly useful for social media and informal text.
- Handles punctuation, capitalization, emojis and intensifiers well.

Pipeline:
Text
  ↓
TextBlob / VADER
  ↓
Sentiment Score
  ↓
Positive / Negative / Neutral
""")
