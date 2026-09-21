# ============================================================
# TOPIC MODELING USING LATENT SEMANTIC ANALYSIS (LSA)
# Google Colab - Complete Program in One Cell
# ============================================================

# Install required library
!pip install -q scikit-learn nltk

# Import libraries
import nltk
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Download stopwords
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))


# ------------------------------------------------------------
# 1. Sample Document Collection
# ------------------------------------------------------------

documents = [
    "Artificial intelligence and machine learning are transforming technology.",
    "Deep learning uses neural networks for artificial intelligence applications.",
    "Machine learning algorithms analyze data and make predictions.",

    "Football is a popular sport played by teams around the world.",
    "The football team won the championship after an exciting match.",
    "Players practice every day to improve their football skills.",

    "Python is a popular programming language for software development.",
    "Software developers use Python to build applications and websites.",
    "Programming languages help developers create efficient software systems."
]


# ------------------------------------------------------------
# 2. Text Preprocessing
# ------------------------------------------------------------

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)


cleaned_documents = [
    preprocess_text(doc)
    for doc in documents
]


print("=" * 70)
print("PREPROCESSED DOCUMENTS")
print("=" * 70)

for i, doc in enumerate(cleaned_documents, 1):
    print(f"Document {i}: {doc}")


# ------------------------------------------------------------
# 3. Convert Documents into TF-IDF Matrix
# ------------------------------------------------------------

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(cleaned_documents)

print("\n" + "=" * 70)
print("TF-IDF MATRIX")
print("=" * 70)

print(tfidf_matrix.toarray())

print("\nVocabulary:")
print(vectorizer.get_feature_names_out())


# ------------------------------------------------------------
# 4. Apply LSA using Truncated SVD
# ------------------------------------------------------------

# Number of hidden topics
num_topics = 3

lsa_model = TruncatedSVD(
    n_components=num_topics,
    random_state=42
)

lsa_matrix = lsa_model.fit_transform(tfidf_matrix)


# ------------------------------------------------------------
# 5. Display Discovered Topics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DISCOVERED TOPICS USING LSA")
print("=" * 70)

feature_names = vectorizer.get_feature_names_out()

for topic_index, topic in enumerate(lsa_model.components_):

    # Get indices of top words
    top_word_indices = topic.argsort()[-8:][::-1]

    top_words = [
        feature_names[i]
        for i in top_word_indices
    ]

    print(f"\nTopic {topic_index + 1}:")
    print(", ".join(top_words))


# ------------------------------------------------------------
# 6. Display Document Representation in Topic Space
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DOCUMENT REPRESENTATION IN LSA TOPIC SPACE")
print("=" * 70)

for i, representation in enumerate(lsa_matrix, 1):

    print(f"\nDocument {i}:")
    print(documents[i - 1])

    for topic_number, value in enumerate(representation, 1):
        print(
            f"  Topic {topic_number}: "
            f"{value:.3f}"
        )


# ------------------------------------------------------------
# 7. Find Dominant Topic for Each Document
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DOMINANT TOPIC FOR EACH DOCUMENT")
print("=" * 70)

for i, representation in enumerate(lsa_matrix, 1):

    dominant_topic = representation.argmax() + 1

    print(
        f"Document {i} → "
        f"Dominant Topic: Topic {dominant_topic}"
    )


# ------------------------------------------------------------
# 8. Explained Variance
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPLAINED VARIANCE BY LSA TOPICS")
print("=" * 70)

for i, variance in enumerate(
    lsa_model.explained_variance_ratio_, 1
):
    print(
        f"Topic {i}: "
        f"{variance:.4f} "
        f"({variance * 100:.2f}%)"
    )

print(
    "\nTotal explained variance: "
    f"{lsa_model.explained_variance_ratio_.sum() * 100:.2f}%"
)


# ------------------------------------------------------------
# 9. LSA Pipeline
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LSA TOPIC MODELING PIPELINE")
print("=" * 70)

print("""
Documents
    ↓
Text Preprocessing
    ↓
TF-IDF Matrix
    ↓
Singular Value Decomposition (SVD)
    ↓
Reduced Semantic Space
    ↓
Hidden Topics
    ↓
Important Topic Words
""")

print("""
LSA = Latent Semantic Analysis

LSA discovers hidden semantic topics by reducing
the dimensionality of the TF-IDF document-term matrix.

Example:

Topic 1 → AI, machine learning, neural networks
Topic 2 → football, team, players, championship
Topic 3 → Python, programming, software, developers
""")
