# Text Similarity using Word Mover's Distance (WMD)
# Using Gensim + Pre-trained GloVe Embeddings

# ============================================================
# 1. INSTALL REQUIRED LIBRARIES
# ============================================================

!pip install -q gensim==4.3.3
!pip install -q scipy==1.13.1
!wget -q https://nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import numpy as np
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess

print("=" * 70)
print("TEXT SIMILARITY USING WORD MOVER'S DISTANCE (WMD)")
print("=" * 70)


# ============================================================
# 3. LOAD GLOVE EMBEDDINGS
# ============================================================

print("\nLoading GloVe embeddings...")

glove_file = "glove.6B.100d.txt"

model = KeyedVectors.load_word2vec_format(
    glove_file,
    binary=False,
    no_header=True
)

print("GloVe embeddings loaded successfully!")
print("Vector size:", model.vector_size)


# ============================================================
# 4. DEFINE TWO DOCUMENTS
# ============================================================

document1 = """
Machine learning is a field of artificial intelligence.
It allows computers to learn from data.
"""

document2 = """
Artificial intelligence enables computers to learn
using data and machine learning techniques.
"""

document3 = """
I enjoy playing football and watching cricket matches.
"""


print("\n--- DOCUMENTS ---")

print("\nDocument 1:")
print(document1)

print("\nDocument 2:")
print(document2)

print("\nDocument 3:")
print(document3)


# ============================================================
# 5. TOKENIZE DOCUMENTS
# ============================================================

tokens1 = simple_preprocess(document1)
tokens2 = simple_preprocess(document2)
tokens3 = simple_preprocess(document3)

print("\n--- TOKENIZED DOCUMENTS ---")

print("Document 1:", tokens1)
print("Document 2:", tokens2)
print("Document 3:", tokens3)


# ============================================================
# 6. REMOVE WORDS NOT PRESENT IN GLOVE
# ============================================================

tokens1 = [word for word in tokens1 if word in model.key_to_index]
tokens2 = [word for word in tokens2 if word in model.key_to_index]
tokens3 = [word for word in tokens3 if word in model.key_to_index]


# ============================================================
# 7. CALCULATE WMD
# ============================================================

print("\n--- WORD MOVER'S DISTANCE ---")

distance_1_2 = model.wmdistance(tokens1, tokens2)
distance_1_3 = model.wmdistance(tokens1, tokens3)
distance_2_3 = model.wmdistance(tokens2, tokens3)

print(f"Document 1 vs Document 2: {distance_1_2:.4f}")
print(f"Document 1 vs Document 3: {distance_1_3:.4f}")
print(f"Document 2 vs Document 3: {distance_2_3:.4f}")


# ============================================================
# 8. INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""
WMD measures the semantic distance between two documents.

Smaller WMD:
    -> Documents are more semantically similar.

Larger WMD:
    -> Documents are more semantically different.

Example:

Document 1:
Machine learning + Artificial Intelligence

Document 2:
Artificial intelligence + Machine learning

These documents discuss similar concepts,
so their WMD should be relatively small.

Document 3:
Football + Cricket

This discusses a different topic,
so its WMD with Documents 1 and 2 should be larger.

Pipeline:

Documents
    ↓
Tokenization
    ↓
GloVe Word Embeddings
    ↓
Word Mover's Distance
    ↓
Semantic Distance
""")


# ============================================================
# 9. FINAL RESULT
# ============================================================

print("=" * 70)
print("WMD COMPUTATION COMPLETED")
print("=" * 70)
