# Named Entity Recognition (NER) using spaCy

# Install spaCy
!pip install -q spacy
!python -m spacy download en_core_web_sm -q

# Import spaCy
import spacy

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Given text
text = """
Aditi is studying Computer Science at Delhi University.
She visited New Delhi on 15 August 2026.
Microsoft announced a new AI project worth 5 million dollars.
"""

print("=" * 70)
print("NAMED ENTITY RECOGNITION (NER) USING spaCy")
print("=" * 70)

# Process text
doc = nlp(text)

# Display entities
print("\n--- IDENTIFIED NAMED ENTITIES ---")
print("-" * 50)

for entity in doc.ents:
    print(f"Entity: {entity.text:<25} Label: {entity.label_:<10} Meaning: {spacy.explain(entity.label_)}")


# Display tokens and their entity information
print("\n--- TOKEN + ENTITY INFORMATION ---")
print("-" * 60)

for token in doc:
    if token.ent_type_:
        print(f"{token.text:<20} {token.ent_type_}")


# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
NER identifies important named entities in text.

Common Entity Types:
PERSON       -> Person's name
ORG          -> Organization
GPE          -> Country, city, state, etc.
DATE         -> Date
MONEY        -> Monetary value
LOC          -> Location
PRODUCT      -> Product
EVENT        -> Event

Example:
Microsoft              -> ORG
New Delhi              -> GPE
15 August 2026         -> DATE
5 million dollars     -> MONEY
""")
