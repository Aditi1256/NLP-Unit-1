# Tokenization of Sentences and Words using NLTK and spaCy

# Install libraries and spaCy English model
!pip install -q nltk spacy
!python -m spacy download en_core_web_sm -q

# Import libraries
import nltk
import spacy

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Input text
text = """Natural Language Processing is a part of Artificial Intelligence.
It helps computers understand human language.
NLTK and spaCy are popular Python libraries for NLP."""

print("=" * 60)
print("TOKENIZATION OF SENTENCES AND WORDS")
print("=" * 60)

# ---------------- NLTK ----------------
print("\n--- NLTK ---")

# Sentence Tokenization
print("\nSentence Tokenization:")
nltk_sentences = sent_tokenize(text)

for i, sentence in enumerate(nltk_sentences, 1):
    print(f"{i}. {sentence}")

# Word Tokenization
print("\nWord Tokenization:")
nltk_words = word_tokenize(text)
print(nltk_words)


# ---------------- spaCy ----------------
print("\n--- spaCy ---")

# Process text using spaCy
doc = nlp(text)

# Sentence Tokenization
print("\nSentence Tokenization:")
for i, sentence in enumerate(doc.sents, 1):
    print(f"{i}. {sentence.text}")

# Word Tokenization
print("\nWord Tokenization:")
spacy_words = [token.text for token in doc]
print(spacy_words)
