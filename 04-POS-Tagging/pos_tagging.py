# Part-of-Speech (POS) Tagging using NLTK and spaCy

# Install required libraries
!pip install -q nltk spacy
!python -m spacy download en_core_web_sm -q

# Import libraries
import nltk
import spacy

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Given sentence
sentence = "The students are studying Natural Language Processing."

print("=" * 70)
print("PART-OF-SPEECH (POS) TAGGING")
print("=" * 70)

# ============================================================
# 1. POS TAGGING USING NLTK
# ============================================================

print("\n--- NLTK POS TAGGING ---")

# Tokenize sentence
words = word_tokenize(sentence)

# POS tagging
nltk_tags = pos_tag(words)

print("\nWord\t\tPOS Tag")
print("-" * 30)

for word, tag in nltk_tags:
    print(f"{word:<15} {tag}")


# ============================================================
# 2. POS TAGGING USING spaCy
# ============================================================

print("\n--- spaCy POS TAGGING ---")

# Process sentence
doc = nlp(sentence)

print("\nWord\t\tPOS\t\tDescription")
print("-" * 50)

for token in doc:
    print(f"{token.text:<15} {token.pos_:<10} {token.tag_}")


# ============================================================
# 3. DISPLAY POS TAG MEANINGS
# ============================================================

print("\n--- COMMON POS TAGS ---")

print("""
NN   = Noun
NNS  = Plural Noun
VB   = Verb
VBD  = Past Tense Verb
VBG  = Verb, Gerund/Present Participle
VBN  = Past Participle
JJ   = Adjective
RB   = Adverb
PRP  = Personal Pronoun
DT   = Determiner
IN   = Preposition
CC   = Conjunction
CD   = Cardinal Number
""")

# ============================================================
# 4. SUMMARY
# ============================================================

print("=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
POS Tagging assigns a grammatical category to each word.

Example:
The       -> Determiner
students  -> Noun
are       -> Verb
studying  -> Verb
Natural   -> Adjective
Language  -> Noun
Processing-> Noun

POS tagging is useful for:
- Text analysis
- Grammar analysis
- Named Entity Recognition
- Information Extraction
- NLP applications
""")
