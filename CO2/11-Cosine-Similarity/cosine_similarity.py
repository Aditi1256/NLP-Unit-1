# COSINE SIMILARITY BETWEEN TEXT DOCUMENTS
# Using TF-IDF and Cosine Similarity

# Install required library
!pip install -q scikit-learn

# Import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 70)
print("COSINE SIMILARITY BETWEEN TEXT DOCUMENTS")
print("=" * 70)

# ============================================================
# 1. DOCUMENTS
# ============================================================

documents = [
    "I love machine learning and artificial intelligence",
    "Machine learning is a part of artificial intelligence",
    "I enjoy playing football and cricket"
]

print("\n--- ORIGINAL DOCUMENTS ---")

for i, doc in enumerate(documents, 1):
    print(f"Document {i}: {doc}")


# ============================================================
# 2. CONVERT DOCUMENTS INTO TF-IDF VECTORS
# ============================================================

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

print("\n--- TF-IDF VECTORS ---")

print(tfidf_matrix.toarray())


# ============================================================
# 3. DISPLAY VOCABULARY
# ============================================================

print("\n--- VOCABULARY ---")

print(vectorizer.get_feature_names_out())


# ============================================================
# 4. CALCULATE COSINE SIMILARITY
# ============================================================

similarity_matrix = cosine_similarity(tfidf_matrix)

print("\n--- COSINE SIMILARITY MATRIX ---")

print(similarity_matrix)


# ============================================================
# 5. DOCUMENT-WISE COMPARISON
# ============================================================

print("\n--- DOCUMENT SIMILARITY ---")

for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        similarity = similarity_matrix[i][j]

        print(
            f"Document {i + 1} vs Document {j + 1} "
            f"= {similarity:.4f}"
        )


# ============================================================
# 6. INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""
Cosine Similarity measures how similar two text documents are.

Range:
0  -> Completely different
1  -> Identical / highly similar

Example:
Document 1 and Document 2 contain many common words
related to machine learning and artificial intelligence,
so their similarity will be relatively high.

Document 3 talks about football and cricket,
so its similarity with Documents 1 and 2 will be low.

Steps:
1. Convert text into TF-IDF vectors.
2. Calculate cosine similarity between vectors.
3. Compare the similarity scores.
""")
