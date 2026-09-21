# Parsing and Chunking using RegEx and spaCy

# Install required libraries
!pip install -q nltk spacy
!python -m spacy download en_core_web_sm -q

# Import libraries
import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Given sentence
sentence = "The intelligent student is studying Natural Language Processing."

print("=" * 70)
print("PARSING AND CHUNKING USING REGEX AND SPACY")
print("=" * 70)


# ============================================================
# 1. REGEX CHUNKING USING NLTK
# ============================================================

print("\n--- 1. CHUNKING USING REGEX + NLTK ---")

# Tokenization
words = word_tokenize(sentence)

# POS Tagging
pos_tags = pos_tag(words)

print("\nPOS Tagged Sentence:")
print(pos_tags)

# Define Regex grammar
grammar = r"""
    NP: {<DT>?<JJ>*<NN|NNS>+}
"""

# Create Regex Parser
chunk_parser = RegexpParser(grammar)

# Parse the sentence
tree = chunk_parser.parse(pos_tags)

print("\nChunked Tree:")
print(tree)

print("\nNoun Phrases:")
for subtree in tree.subtrees():
    if subtree.label() == "NP":
        print(" ".join(word for word, tag in subtree.leaves()))


# ============================================================
# 2. CHUNKING USING spaCy
# ============================================================

print("\n--- 2. CHUNKING USING spaCy ---")

# Process sentence
doc = nlp(sentence)

print("\nTokens and POS Tags:")
print("-" * 40)

for token in doc:
    print(f"{token.text:<20} {token.pos_}")

print("\nNoun Phrases using spaCy:")
print("-" * 40)

for chunk in doc.noun_chunks:
    print(f"Text: {chunk.text}")
    print(f"Root: {chunk.root.text}")
    print(f"Root POS: {chunk.root.pos_}")
    print()


# ============================================================
# 3. DEPENDENCY PARSING USING spaCy
# ============================================================

print("--- 3. DEPENDENCY PARSING USING spaCy ---")

print("\nWord\t\tPOS\t\tDependency\tHead")
print("-" * 60)

for token in doc:
    print(
        f"{token.text:<15}"
        f"{token.pos_:<10}"
        f"{token.dep_:<15}"
        f"{token.head.text}"
    )


# ============================================================
# 4. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Parsing:
Parsing analyzes the grammatical structure and relationships
between words in a sentence.

Chunking:
Chunking groups related words into meaningful phrases.

Regex Chunking:
Uses grammatical patterns based on POS tags.

Example Regex:
NP: {<DT>?<JJ>*<NN|NNS>+}

Meaning:
DT      -> Determiner (optional)
JJ      -> Adjective (zero or more)
NN/NNS  -> Noun

Example:
"The intelligent student"
        ↓
     Noun Phrase (NP)

spaCy:
Uses doc.noun_chunks for noun phrases and
token.dep_ for dependency parsing.
""")
