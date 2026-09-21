# N-GRAM MODEL (Uni-gram, Bi-gram, Tri-gram) GENERATION FROM CORPUS

# Install required library
!pip install -q nltk

# Import libraries
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("=" * 70)
print("N-GRAM MODEL: UNI-GRAM, BI-GRAM AND TRI-GRAM")
print("=" * 70)

# ============================================================
# 1. CORPUS
# ============================================================

corpus = """
Natural language processing is a branch of artificial intelligence.
Natural language processing helps computers understand human language.
Machine learning is used in natural language processing.
"""

print("\n--- CORPUS ---")
print(corpus)


# ============================================================
# 2. TOKENIZATION
# ============================================================

# Convert corpus into words
tokens = word_tokenize(corpus.lower())

# Remove punctuation
tokens = [word for word in tokens if word.isalpha()]

print("\n--- TOKENS ---")
print(tokens)


# ============================================================
# 3. UNIGRAM
# ============================================================

print("\n--- UNIGRAM (1-GRAM) ---")

unigrams = list(ngrams(tokens, 1))

for gram in unigrams:
    print(gram)


# ============================================================
# 4. BIGRAM
# ============================================================

print("\n--- BIGRAM (2-GRAM) ---")

bigrams = list(ngrams(tokens, 2))

for gram in bigrams:
    print(gram)


# ============================================================
# 5. TRIGRAM
# ============================================================

print("\n--- TRIGRAM (3-GRAM) ---")

trigrams = list(ngrams(tokens, 3))

for gram in trigrams:
    print(gram)


# ============================================================
# 6. N-GRAM COUNTS
# ============================================================

from collections import Counter

print("\n--- N-GRAM FREQUENCIES ---")

print("\nUnigram Frequencies:")
unigram_counts = Counter(unigrams)
for gram, count in unigram_counts.items():
    print(gram, ":", count)

print("\nBigram Frequencies:")
bigram_counts = Counter(bigrams)
for gram, count in bigram_counts.items():
    print(gram, ":", count)

print("\nTrigram Frequencies:")
trigram_counts = Counter(trigrams)
for gram, count in trigram_counts.items():
    print(gram, ":", count)


# ============================================================
# 7. SIMPLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
N-Gram:
An N-Gram is a sequence of N consecutive words.

Unigram (1-Gram):
One word at a time.
Example: ("natural",)

Bigram (2-Gram):
Two consecutive words.
Example: ("natural", "language")

Trigram (3-Gram):
Three consecutive words.
Example: ("natural", "language", "processing")

Applications:
- Text prediction
- Autocomplete
- Speech recognition
- Machine translation
- Sentiment analysis
- Language modeling
""")
