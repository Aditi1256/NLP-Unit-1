# GloVe Embeddings Loading and Vector Representation
# Using Pre-trained GloVe 100-dimensional embeddings

# ============================================================
# 1. DOWNLOAD GLOVE EMBEDDINGS
# ============================================================

!wget -q https://nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip

# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import numpy as np

print("=" * 70)
print("GLOVE EMBEDDINGS - LOADING AND VECTOR REPRESENTATION")
print("=" * 70)


# ============================================================
# 3. LOAD GLOVE VECTORS
# ============================================================

embedding_file = "glove.6B.100d.txt"

embeddings = {}

with open(embedding_file, "r", encoding="utf-8") as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embeddings[word] = vector

print("\nGloVe embeddings loaded successfully!")
print("Total words:", len(embeddings))


# ============================================================
# 4. DISPLAY VECTOR REPRESENTATION
# ============================================================

word = "computer"

print(f"\n--- VECTOR REPRESENTATION OF '{word}' ---")

if word in embeddings:
    vector = embeddings[word]

    print("Vector size:", len(vector))
    print("First 20 values:")
    print(vector[:20])

else:
    print("Word not found in GloVe vocabulary.")


# ============================================================
# 5. DISPLAY VECTORS FOR MULTIPLE WORDS
# ============================================================

print("\n--- VECTOR REPRESENTATION OF MULTIPLE WORDS ---")

words = ["machine", "learning", "computer", "language", "artificial"]

for word in words:
    if word in embeddings:
        print(f"\n{word}:")
        print(embeddings[word][:10], "...")

    else:
        print(f"{word}: Not found")


# ============================================================
# 6. COSINE SIMILARITY BETWEEN GLOVE VECTORS
# ============================================================

print("\n--- COSINE SIMILARITY ---")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

word1 = "king"
word2 = "queen"

if word1 in embeddings and word2 in embeddings:

    similarity = cosine_similarity(
        embeddings[word1],
        embeddings[word2]
    )

    print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")

else:
    print("One or both words not found.")


# ============================================================
# 7. WORD ANALOGY
# ============================================================

print("\n--- WORD ANALOGY ---")

# king - man + woman ≈ queen

if all(word in embeddings for word in ["king", "man", "woman", "queen"]):

    analogy_vector = (
        embeddings["king"]
        - embeddings["man"]
        + embeddings["woman"]
    )

    # Calculate similarity with queen
    similarity = cosine_similarity(
        analogy_vector,
        embeddings["queen"]
    )

    print("king - man + woman ≈ queen")
    print("Similarity with queen:", round(similarity, 4))


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
GloVe:
GloVe stands for Global Vectors for Word Representation.

It represents words as numerical vectors based on
word co-occurrence information from a large corpus.

In this program:
1. Pre-trained GloVe vectors are downloaded.
2. 100-dimensional word embeddings are loaded.
3. Individual word vectors are displayed.
4. Cosine similarity is calculated between words.
5. A simple word analogy is demonstrated.

Example:
king -> numerical vector
queen -> numerical vector

Similar words generally have similar vector representations.
""")
