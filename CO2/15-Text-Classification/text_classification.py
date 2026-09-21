# TEXT CLASSIFICATION USING NAIVE BAYES AND SVM WITH TF-IDF

# Install required library
!pip install -q scikit-learn

# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

print("=" * 70)
print("TEXT CLASSIFICATION USING TF-IDF + NAIVE BAYES + SVM")
print("=" * 70)


# ============================================================
# 1. CUSTOM DATASET
# ============================================================

texts = [
    "I love this movie it is amazing",
    "This movie was excellent and wonderful",
    "The film was fantastic and very enjoyable",
    "I really liked this movie",
    "The acting was great and impressive",
    "This is a beautiful and entertaining movie",
    "I hate this movie it is terrible",
    "This movie was boring and disappointing",
    "The film was horrible and waste of time",
    "I did not like this movie",
    "The acting was bad and disappointing",
    "This is a terrible and boring movie"
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative"
]

print("\n--- DATASET ---")

for text, label in zip(texts, labels):
    print(f"{label:<10} -> {text}")


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 3. TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer()

# Learn vocabulary from training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data using same vocabulary
X_test_tfidf = vectorizer.transform(X_test)

print("\n--- TF-IDF ---")

print("Number of features:", len(vectorizer.get_feature_names_out()))

print("\nTF-IDF Feature Names:")
print(vectorizer.get_feature_names_out())

print("\nTraining TF-IDF Matrix:")
print(X_train_tfidf.toarray())


# ============================================================
# 4. NAIVE BAYES CLASSIFIER
# ============================================================

print("\n--- NAIVE BAYES CLASSIFIER ---")

nb_model = MultinomialNB()

# Train model
nb_model.fit(X_train_tfidf, y_train)

# Predict
nb_predictions = nb_model.predict(X_test_tfidf)

# Accuracy
nb_accuracy = accuracy_score(y_test, nb_predictions)

print("Naive Bayes Accuracy:", round(nb_accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, nb_predictions))


# ============================================================
# 5. SVM CLASSIFIER
# ============================================================

print("\n--- SVM CLASSIFIER ---")

svm_model = LinearSVC()

# Train model
svm_model.fit(X_train_tfidf, y_train)

# Predict
svm_predictions = svm_model.predict(X_test_tfidf)

# Accuracy
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("SVM Accuracy:", round(svm_accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, svm_predictions))


# ============================================================
# 6. COMPARE MODELS
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(f"Naive Bayes Accuracy : {nb_accuracy:.4f}")
print(f"SVM Accuracy         : {svm_accuracy:.4f}")


# ============================================================
# 7. PREDICT NEW TEXT
# ============================================================

print("\n--- NEW TEXT PREDICTION ---")

new_texts = [
    "This movie was amazing and wonderful",
    "This movie was boring and terrible"
]

new_text_tfidf = vectorizer.transform(new_texts)

nb_new_predictions = nb_model.predict(new_text_tfidf)
svm_new_predictions = svm_model.predict(new_text_tfidf)

for i, text in enumerate(new_texts):

    print("\nText:", text)

    print("Naive Bayes Prediction:", nb_new_predictions[i])
    print("SVM Prediction:", svm_new_predictions[i])


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Text Classification Pipeline:

Text
 ↓
TF-IDF Vectorization
 ↓
Numerical Feature Vectors
 ↓
 ┌─────────────────┐
 │                 │
Naive Bayes       SVM
 │                 │
 └────────┬────────┘
          ↓
    Classification

Naive Bayes:
- Probabilistic classifier
- Based on Bayes theorem
- Fast and simple
- Commonly used for text classification

SVM:
- Finds a decision boundary between classes
- LinearSVC is commonly used for text classification
- Works well with high-dimensional TF-IDF features

TF-IDF:
Converts text into numerical features based on
the importance of words in documents.
""")
