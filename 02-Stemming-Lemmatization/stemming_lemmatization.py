# NLP PROGRAM:
# Tokenization, Stemming and Lemmatization using NLTK and spaCy

# Install required libraries
!pip install -q nltk spacy
!python -m spacy download en_core_web_sm -q

# Import libraries
import nltk
import spacy

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Input text
text = """The students are studying natural language processing.
They are learning about computers, languages and programming.
The students studied many interesting topics."""

print("=" * 70)
print("TOKENIZATION, STEMMING AND LEMMATIZATION")
print("=" * 70)


# ============================================================
# 1. SENTENCE TOKENIZATION USING NLTK
# ============================================================

print("\n--- 1. NLTK SENTENCE TOKENIZATION ---")

sentences = sent_tokenize(text)

for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")


# ============================================================
# 2. WORD TOKENIZATION USING NLTK
# ============================================================

print("\n--- 2. NLTK WORD TOKENIZATION ---")

words = word_tokenize(text)
print(words)


# ============================================================
# 3. STEMMING USING NLTK
# ============================================================

print("\n--- 3. NLTK STEMMING ---")

stemmer = PorterStemmer()

# Only words, ignoring punctuation
for word in words:
    if word.isalpha():
        print(f"{word} -> {stemmer.stem(word)}")


# ============================================================
# 4. LEMMATIZATION USING NLTK
# ============================================================

print("\n--- 4. NLTK LEMMATIZATION ---")

lemmatizer = WordNetLemmatizer()

for word in words:
    if word.isalpha():
        print(f"{word} -> {lemmatizer.lemmatize(word)}")


# ============================================================
# 5. SENTENCE TOKENIZATION USING spaCy
# ============================================================

print("\n--- 5. spaCy SENTENCE TOKENIZATION ---")

doc = nlp(text)

for i, sentence in enumerate(doc.sents, 1):
    print(f"{i}. {sentence.text}")


# ============================================================
# 6. WORD TOKENIZATION USING spaCy
# ============================================================

print("\n--- 6. spaCy WORD TOKENIZATION ---")

for token in doc:
    print(token.text)


# ============================================================
# 7. LEMMATIZATION USING spaCy
# ============================================================

print("\n--- 7. spaCy LEMMATIZATION ---")

for token in doc:
    if token.is_alpha:
        print(f"{token.text} -> {token.lemma_}")


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Tokenization:
Breaks text into sentences or individual words.

Stemming:
Cuts words to their basic/root-like form.
Example: studying -> studi

Lemmatization:
Converts a word to its meaningful dictionary base form.
Example: studying -> study
""")
