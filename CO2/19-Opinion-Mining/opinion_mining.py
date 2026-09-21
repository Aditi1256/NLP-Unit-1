# ============================================================
# OPINION MINING ON PRODUCT / SERVICE REVIEWS DATASET
# Google Colab - Complete Program in One Cell
# ============================================================

# Install required libraries
!pip install -q nltk pandas matplotlib

# Import libraries
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon', quiet=True)

# Create sentiment analyzer
sia = SentimentIntensityAnalyzer()


# ------------------------------------------------------------
# 1. Create Product / Service Reviews Dataset
# ------------------------------------------------------------

reviews = [
    "The product quality is excellent and I really love it.",
    "Amazing product, very useful and easy to use.",
    "The delivery was fast and the packaging was perfect.",
    "I am very happy with this product.",
    "Excellent service and friendly customer support.",
    
    "The product is terrible and stopped working after two days.",
    "Very poor quality and completely disappointing.",
    "The delivery was extremely late.",
    "I hate this product. It is a complete waste of money.",
    "The customer service was rude and unhelpful.",
    
    "The product is okay, nothing special.",
    "The service was average.",
    "Delivery was neither fast nor slow.",
    "The product works as expected.",
    "It is an ordinary product for the price."
]


# ------------------------------------------------------------
# 2. Create DataFrame
# ------------------------------------------------------------

df = pd.DataFrame({
    "Review": reviews
})

print("=" * 70)
print("PRODUCT / SERVICE REVIEWS DATASET")
print("=" * 70)

print(df.to_string(index=False))


# ------------------------------------------------------------
# 3. Function for Opinion Mining
# ------------------------------------------------------------

def analyze_sentiment(text):

    scores = sia.polarity_scores(text)

    compound = scores["compound"]

    # Classify opinion
    if compound >= 0.05:
        opinion = "Positive"
    elif compound <= -0.05:
        opinion = "Negative"
    else:
        opinion = "Neutral"

    return pd.Series([
        scores["pos"],
        scores["neg"],
        scores["neu"],
        compound,
        opinion
    ])


# ------------------------------------------------------------
# 4. Analyze Every Review
# ------------------------------------------------------------

df[
    ["Positive Score",
     "Negative Score",
     "Neutral Score",
     "Compound Score",
     "Opinion"]
] = df["Review"].apply(analyze_sentiment)


# ------------------------------------------------------------
# 5. Display Results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OPINION MINING RESULTS")
print("=" * 70)

for index, row in df.iterrows():

    print(f"\nReview {index + 1}:")
    print(row["Review"])

    print(f"Positive Score : {row['Positive Score']:.2f}")
    print(f"Negative Score : {row['Negative Score']:.2f}")
    print(f"Neutral Score  : {row['Neutral Score']:.2f}")
    print(f"Compound Score : {row['Compound Score']:.2f}")
    print(f"Opinion        : {row['Opinion']}")

    print("-" * 70)


# ------------------------------------------------------------
# 6. Overall Opinion Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL OPINION SUMMARY")
print("=" * 70)

opinion_counts = df["Opinion"].value_counts()

print(opinion_counts)


# ------------------------------------------------------------
# 7. Calculate Opinion Percentages
# ------------------------------------------------------------

opinion_percentage = (
    df["Opinion"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nOpinion Percentage:")
print(opinion_percentage)


# ------------------------------------------------------------
# 8. Find Average Sentiment Scores
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AVERAGE SENTIMENT SCORES")
print("=" * 70)

print(
    "Average Positive Score:",
    round(df["Positive Score"].mean(), 3)
)

print(
    "Average Negative Score:",
    round(df["Negative Score"].mean(), 3)
)

print(
    "Average Neutral Score:",
    round(df["Neutral Score"].mean(), 3)
)

print(
    "Average Compound Score:",
    round(df["Compound Score"].mean(), 3)
)


# ------------------------------------------------------------
# 9. Visualization
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

opinion_counts.plot(kind="bar")

plt.title("Product / Service Opinion Distribution")
plt.xlabel("Opinion")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)

plt.show()


# ------------------------------------------------------------
# 10. Identify Most Positive and Most Negative Reviews
# ------------------------------------------------------------

most_positive = df.loc[
    df["Compound Score"].idxmax()
]

most_negative = df.loc[
    df["Compound Score"].idxmin()
]

print("\n" + "=" * 70)
print("MOST POSITIVE REVIEW")
print("=" * 70)

print(most_positive["Review"])
print("Compound Score:",
      round(most_positive["Compound Score"], 3))


print("\n" + "=" * 70)
print("MOST NEGATIVE REVIEW")
print("=" * 70)

print(most_negative["Review"])
print("Compound Score:",
      round(most_negative["Compound Score"], 3))


# ------------------------------------------------------------
# 11. Final Opinion Mining Pipeline
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OPINION MINING PIPELINE")
print("=" * 70)

print("""
Product / Service Reviews
          ↓
      Text Data
          ↓
   VADER Sentiment Analysis
          ↓
Positive / Negative / Neutral Scores
          ↓
     Compound Score
          ↓
Positive / Negative / Neutral Opinion
          ↓
    Overall Opinion Summary
""")
