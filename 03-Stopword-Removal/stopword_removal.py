# Stop-word Removal from a Document using NLTK and spaCy

# Install required libraries
!pip install -q nltk spacy
!python -m spacy download en_core_web_sm -q

# Import libraries
import nltk
import spacy

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Input document
text = """
Natural Language Processing is a branch of Artificial Intelligence.
It helps computers understand and process human language.
The main goal is to make computers understand text in a useful way.
"""

print("=" * 70)
print("STOP-WORD REMOVAL FROM A DOCUMENT")
print("=" * 70)

# ============================================================
# 1. NLTK STOP-WORD REMOVAL
# ============================================================

print("\n--- NLTK STOP-WORD REMOVAL ---")

# Convert text into words
words = word_tokenize(text)

# Get English stop words
stop_words = set(stopwords.words('english'))

# Remove stop words
filtered_words = [
    word for word in words
    if word.lower() not in stop_words and word.isalpha()
]

print("\nOriginal Words:")
print([word for word in words if word.isalpha()])

print("\nStop Words Removed:")
print(filtered_words)

print("\nFinal Document:")
print(" ".join(filtered_words))


# ============================================================
# 2. spaCy STOP-WORD REMOVAL
# ============================================================

print("\n--- spaCy STOP-WORD REMOVAL ---")

# Process document
doc = nlp(text)

# Remove stop words and punctuation
filtered_spacy = [
    token.text
    for token in doc
    if not token.is_stop and not token.is_punct and token.is_alpha
]

print("\nOriginal Words:")
print([token.text for token in doc if token.is_alpha])

print("\nStop Words Removed:")
print(filtered_spacy)

print("\nFinal Document:")
print(" ".join(filtered_spacy))


# ============================================================
# 3. DISPLAY SOME STOP WORDS
# ============================================================

print("\n--- EXAMPLE STOP WORDS ---")

print("NLTK Stop Words:")
print(list(stop_words)[:30])

print("\nTotal NLTK Stop Words:", len(stop_words))


# ============================================================
# 4. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Stop-word removal removes common words that usually carry
less useful information for many NLP tasks.

Examples:
the, is, a, an, and, of, to, in, for, etc.

NLTK:
Uses stopwords.words('english')

spaCy:
Uses token.is_stop

Example:
"The students are studying NLP"

After stop-word removal:
"students studying NLP"
""")
