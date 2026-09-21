# Bag-of-Words (BoW) Vectorization and Representation
# Using CountVectorizer from scikit-learn

# Install required library
!pip install -q scikit-learn

# Import libraries
from sklearn.feature_extraction.text import CountVectorizer

print("=" * 70)
print("BAG-OF-WORDS (BoW) VECTORIZATION AND REPRESENTATION")
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
# 2. CREATE BAG-OF-WORDS MODEL
# ============================================================

vectorizer = CountVectorizer()

# Learn vocabulary and transform documents into vectors
bow_matrix = vectorizer.fit_transform(documents)


# ============================================================
# 3. DISPLAY VOCABULARY
# ============================================================

print("\n--- VOCABULARY ---")

vocabulary = vectorizer.get_feature_names_out()

print(vocabulary)


# ============================================================
# 4. DISPLAY BoW VECTORS
# ============================================================

print("\n--- BAG-OF-WORDS VECTORS ---")

print(bow_matrix.toarray())


# ============================================================
# 5. DISPLAY DOCUMENT-WISE REPRESENTATION
# ============================================================

print("\n--- DOCUMENT REPRESENTATION ---")

for i, vector in enumerate(bow_matrix.toarray(), 1):
    print(f"Document {i}:")
    print(vector)


# ============================================================
# 6. DISPLAY WORD + COUNT
# ============================================================

print("\n--- WORD FREQUENCY IN EACH DOCUMENT ---")

for i, vector in enumerate(bow_matrix.toarray(), 1):
    print(f"\nDocument {i}:")
    
    for word, count in zip(vocabulary, vector):
        if count > 0:
            print(f"{word:<20} : {count}")


# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Bag-of-Words (BoW):
BoW converts text into numerical vectors based on word frequency.

Steps:
1. Collect documents
2. Create vocabulary of unique words
3. Count how many times each word occurs
4. Represent each document as a numerical vector

Example:

Documents:
"cat dog"
"cat cat dog"

Vocabulary:
[cat, dog]

Vectors:
"cat dog"       -> [1, 1]
"cat cat dog"   -> [2, 1]

BoW does NOT consider:
- Word order
- Grammar
- Context
- Meaning

It only considers word occurrence/frequency.
""")
