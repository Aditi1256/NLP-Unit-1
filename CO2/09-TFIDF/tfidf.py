# TF-IDF Implementation and Comparison with Bag-of-Words (BoW)

# Install required library
!pip install -q scikit-learn

# Import libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

print("=" * 70)
print("TF-IDF IMPLEMENTATION AND COMPARISON WITH BoW")
print("=" * 70)

# ============================================================
# 1. DOCUMENTS
# ============================================================

documents = [
    "I love natural language processing",
    "I love machine learning",
    "Natural language processing is interesting"
]

print("\n--- ORIGINAL DOCUMENTS ---")

for i, doc in enumerate(documents, 1):
    print(f"Document {i}: {doc}")


# ============================================================
# 2. BAG-OF-WORDS (BoW)
# ============================================================

print("\n--- BAG-OF-WORDS (BoW) ---")

bow_vectorizer = CountVectorizer()

bow_matrix = bow_vectorizer.fit_transform(documents)

bow_features = bow_vectorizer.get_feature_names_out()

print("\nVocabulary:")
print(bow_features)

print("\nBoW Matrix:")
print(bow_matrix.toarray())


# ============================================================
# 3. TF-IDF
# ============================================================

print("\n--- TF-IDF ---")

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_features = tfidf_vectorizer.get_feature_names_out()

print("\nVocabulary:")
print(tfidf_features)

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())


# ============================================================
# 4. WORD-WISE TF-IDF VALUES
# ============================================================

print("\n--- TF-IDF WORD VALUES ---")

for i, vector in enumerate(tfidf_matrix.toarray(), 1):
    print(f"\nDocument {i}:")

    for word, value in zip(tfidf_features, vector):
        if value > 0:
            print(f"{word:<20} : {value:.4f}")


# ============================================================
# 5. COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("BoW vs TF-IDF")
print("=" * 70)

print("""
BoW:
- Counts how many times a word appears.
- Common words can get high values.
- Does not consider importance of a word across documents.
- Simple and easy to understand.

TF-IDF:
- Measures how important a word is in a document.
- Reduces the importance of words appearing in many documents.
- Gives higher values to words that are more specific to a document.
- Produces decimal values.

BoW Example:
"love" appears in Document 1 and Document 2
→ counted based on frequency.

TF-IDF:
If a word appears in many documents,
its importance is reduced.

Therefore:
BoW  → Word Frequency
TF-IDF → Word Importance
""")

# ============================================================
# 6. SIDE-BY-SIDE COMPARISON
# ============================================================

print("\n--- SIDE-BY-SIDE COMPARISON ---")

print("\nBoW Representation:")
print(bow_matrix.toarray())

print("\nTF-IDF Representation:")
print(tfidf_matrix.toarray())
