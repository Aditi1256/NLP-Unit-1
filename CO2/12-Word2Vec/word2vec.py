# Word2Vec Word Embeddings using Gensim on a Custom Corpus

# Install required libraries
!pip install -q gensim nltk

# Import libraries
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("=" * 70)
print("WORD2VEC WORD EMBEDDINGS USING GENSIM")
print("=" * 70)

# ============================================================
# 1. CUSTOM CORPUS
# ============================================================

corpus = [
    "machine learning is a part of artificial intelligence",
    "deep learning is a type of machine learning",
    "artificial intelligence is changing the world",
    "machine learning helps computers learn from data",
    "deep learning uses neural networks",
    "neural networks are useful for deep learning",
    "natural language processing is an application of artificial intelligence",
    "machine learning is used in natural language processing",
    "artificial intelligence and machine learning are important technologies",
    "data science uses machine learning techniques"
]

print("\n--- CUSTOM CORPUS ---")

for i, sentence in enumerate(corpus, 1):
    print(f"{i}. {sentence}")


# ============================================================
# 2. TOKENIZE CORPUS
# ============================================================

tokenized_corpus = [
    word_tokenize(sentence.lower())
    for sentence in corpus
]

print("\n--- TOKENIZED CORPUS ---")

for sentence in tokenized_corpus:
    print(sentence)


# ============================================================
# 3. TRAIN WORD2VEC MODEL
# ============================================================

model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1
)

print("\n--- WORD2VEC MODEL TRAINED SUCCESSFULLY ---")


# ============================================================
# 4. DISPLAY WORD VECTOR
# ============================================================

word = "machine"

print(f"\n--- VECTOR REPRESENTATION OF '{word}' ---")

vector = model.wv[word]

print("Vector size:", len(vector))
print("Vector:")
print(vector)


# ============================================================
# 5. FIND SIMILAR WORDS
# ============================================================

print("\n--- WORDS SIMILAR TO 'machine' ---")

similar_words = model.wv.most_similar("machine", topn=5)

for word, similarity in similar_words:
    print(f"{word:<20} : {similarity:.4f}")


# ============================================================
# 6. FIND SIMILAR WORDS TO 'learning'
# ============================================================

print("\n--- WORDS SIMILAR TO 'learning' ---")

similar_words = model.wv.most_similar("learning", topn=5)

for word, similarity in similar_words:
    print(f"{word:<20} : {similarity:.4f}")


# ============================================================
# 7. WORD SIMILARITY
# ============================================================

print("\n--- WORD SIMILARITY ---")

word1 = "machine"
word2 = "learning"

similarity = model.wv.similarity(word1, word2)

print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")


# ============================================================
# 8. CHECK VOCABULARY
# ============================================================

print("\n--- WORD2VEC VOCABULARY ---")

vocabulary = list(model.wv.key_to_index.keys())

print(vocabulary)


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Word2Vec:
Word2Vec converts words into numerical vectors called embeddings.

Words appearing in similar contexts tend to have similar vectors.

Important parameters:

vector_size = 100
    Size of each word vector.

window = 5
    Number of surrounding words considered.

min_count = 1
    Minimum frequency required for a word.

sg = 1
    Uses Skip-Gram architecture.

Main operations:

model.wv[word]
    -> Gets the vector of a word.

model.wv.most_similar(word)
    -> Finds words with similar embeddings.

model.wv.similarity(word1, word2)
    -> Calculates similarity between two words.
""")
