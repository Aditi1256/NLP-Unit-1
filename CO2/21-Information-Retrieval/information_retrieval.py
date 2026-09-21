# ============================================================
# INFORMATION RETRIEVAL SYSTEM WITH RANKING USING TF-IDF
# Google Colab - Complete Program in One Cell
# ============================================================

# Install required library
!pip install -q scikit-learn nltk

# Import libraries
import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords', quiet=True)

stop_words = stopwords.words('english')


# ============================================================
# 1. DOCUMENT COLLECTION
# ============================================================

documents = [
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning uses neural networks for machine learning.",
    "Artificial intelligence helps computers solve complex problems.",
    "Python is widely used for machine learning and data science.",
    "Natural language processing helps computers understand human language.",
    "Data science uses statistics machine learning and programming.",
    "Neural networks are important in deep learning applications.",
    "Information retrieval systems search and rank relevant documents."
]


# ============================================================
# 2. DISPLAY DOCUMENTS
# ============================================================

print("=" * 70)
print("DOCUMENT COLLECTION")
print("=" * 70)

for i, document in enumerate(documents, 1):
    print(f"Document {i}: {document}")


# ============================================================
# 3. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text


cleaned_documents = [
    preprocess_text(doc)
    for doc in documents
]


# ============================================================
# 4. CREATE TF-IDF VECTORS
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words=stop_words
)

tfidf_matrix = vectorizer.fit_transform(cleaned_documents)

print("\n" + "=" * 70)
print("TF-IDF MATRIX")
print("=" * 70)

print(tfidf_matrix.toarray())

print("\nVocabulary:")
print(vectorizer.get_feature_names_out())


# ============================================================
# 5. TAKE USER QUERY
# ============================================================

print("\n" + "=" * 70)
print("INFORMATION RETRIEVAL")
print("=" * 70)

query = input(
    "Enter your search query "
    "(Example: machine learning): "
)

query = preprocess_text(query)


# ============================================================
# 6. CONVERT QUERY INTO TF-IDF VECTOR
# ============================================================

query_vector = vectorizer.transform([query])


# ============================================================
# 7. CALCULATE COSINE SIMILARITY
# ============================================================

similarity_scores = cosine_similarity(
    query_vector,
    tfidf_matrix
).flatten()


# ============================================================
# 8. RANK DOCUMENTS
# ============================================================

ranked_indices = np.argsort(
    similarity_scores
)[::-1]


# ============================================================
# 9. DISPLAY RANKED RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RANKED SEARCH RESULTS")
print("=" * 70)

rank = 1

for index in ranked_indices:

    score = similarity_scores[index]

    # Show only documents having some relevance
    if score > 0:

        print(f"\nRank {rank}")
        print(f"Document {index + 1}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Document: {documents[index]}")

        rank += 1


# ============================================================
# 10. SHOW ALL DOCUMENT SCORES
# ============================================================

print("\n" + "=" * 70)
print("ALL DOCUMENT SIMILARITY SCORES")
print("=" * 70)

for index in ranked_indices:

    print(
        f"Document {index + 1}: "
        f"{similarity_scores[index]:.4f}"
    )


# ============================================================
# 11. TOP RESULT
# ============================================================

best_index = ranked_indices[0]

print("\n" + "=" * 70)
print("MOST RELEVANT DOCUMENT")
print("=" * 70)

print("Query:", query)
print("Document:", documents[best_index])
print(
    "Similarity Score:",
    round(similarity_scores[best_index], 4)
)


# ============================================================
# 12. INFORMATION RETRIEVAL PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("INFORMATION RETRIEVAL PIPELINE")
print("=" * 70)

print("""
Document Collection
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
      Index
        ↓
    User Query
        ↓
Convert Query to TF-IDF Vector
        ↓
Cosine Similarity
        ↓
Similarity Scores
        ↓
Rank Documents
        ↓
Most Relevant Documents
""")
