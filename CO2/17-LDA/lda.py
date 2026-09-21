# ============================================================
# TOPIC MODELING USING LATENT DIRICHLET ALLOCATION (LDA)
# Google Colab - Complete Program in One Cell
# ============================================================

# Install required libraries
!pip install -q scikit-learn nltk

# Import libraries
import nltk
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download stopwords
nltk.download('stopwords', quiet=True)

# English stopwords
stop_words = set(stopwords.words('english'))


# ------------------------------------------------------------
# 1. Sample Document Collection
# ------------------------------------------------------------

documents = [
    "Artificial intelligence and machine learning are transforming technology.",
    "Deep learning uses neural networks to solve complex artificial intelligence problems.",
    "Machine learning algorithms can analyze data and make predictions.",
    
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

    # Remove stopwords and very short words
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
# 3. Convert Documents into Bag-of-Words Matrix
# ------------------------------------------------------------

vectorizer = CountVectorizer()

doc_term_matrix = vectorizer.fit_transform(cleaned_documents)

print("\n" + "=" * 70)
print("DOCUMENT-TERM MATRIX")
print("=" * 70)

print(doc_term_matrix.toarray())

print("\nVocabulary:")
print(vectorizer.get_feature_names_out())


# ------------------------------------------------------------
# 4. Apply LDA
# ------------------------------------------------------------

# We know that our sample corpus contains approximately
# 3 major topics:
# 1. AI / Machine Learning
# 2. Football / Sports
# 3. Programming / Software

num_topics = 3

lda_model = LatentDirichletAllocation(
    n_components=num_topics,
    random_state=42
)

lda_model.fit(doc_term_matrix)


# ------------------------------------------------------------
# 5. Display Topics and Important Words
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DISCOVERED TOPICS USING LDA")
print("=" * 70)

feature_names = vectorizer.get_feature_names_out()

for topic_index, topic in enumerate(lda_model.components_):

    # Get indices of top words
    top_word_indices = topic.argsort()[-8:][::-1]

    top_words = [
        feature_names[i]
        for i in top_word_indices
    ]

    print(f"\nTopic {topic_index + 1}:")
    print(", ".join(top_words))


# ------------------------------------------------------------
# 6. Find Dominant Topic for Each Document
# ------------------------------------------------------------

document_topic_distribution = lda_model.transform(doc_term_matrix)

print("\n" + "=" * 70)
print("DOCUMENT-WISE TOPIC DISTRIBUTION")
print("=" * 70)

for i, topic_distribution in enumerate(document_topic_distribution, 1):

    dominant_topic = topic_distribution.argmax() + 1

    print(f"\nDocument {i}:")
    print(documents[i - 1])

    print("Topic probabilities:")

    for topic_number, probability in enumerate(topic_distribution, 1):
        print(
            f"  Topic {topic_number}: "
            f"{probability:.3f}"
        )

    print(f"Dominant Topic: Topic {dominant_topic}")


# ------------------------------------------------------------
# 7. Simple Explanation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LDA TOPIC MODELING PIPELINE")
print("=" * 70)

print("""
Documents
    ↓
Text Preprocessing
    ↓
Remove Stopwords
    ↓
Bag-of-Words / Document-Term Matrix
    ↓
LDA Algorithm
    ↓
Discover Hidden Topics
    ↓
Important Words for Each Topic
    ↓
Assign Dominant Topic to Documents
""")

print("""
LDA = Latent Dirichlet Allocation

LDA is an unsupervised machine learning algorithm used
to discover hidden topics in a collection of documents.

For example:

Topic 1 → AI, machine learning, neural networks
Topic 2 → football, team, players, championship
Topic 3 → Python, programming, software, developers
""")
