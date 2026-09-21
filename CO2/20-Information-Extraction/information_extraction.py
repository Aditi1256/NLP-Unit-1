# ============================================================
# INFORMATION EXTRACTION (IE) FROM
# STRUCTURED / UNSTRUCTURED DOCUMENTS
# Google Colab - Complete Program in One Cell
# ============================================================

# Install required libraries
!pip install -q pandas spacy

# Download spaCy English model
!python -m spacy download en_core_web_sm -q

# Import libraries
import re
import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# ============================================================
# PART 1: INFORMATION EXTRACTION FROM STRUCTURED DATA
# ============================================================

print("=" * 70)
print("PART 1: INFORMATION EXTRACTION FROM STRUCTURED DATA")
print("=" * 70)

# Create a sample structured dataset
data = {
    "Name": [
        "Aditi Gupta",
        "Rahul Sharma",
        "Priya Singh",
        "Arjun Kumar"
    ],
    "Email": [
        "aditi@gmail.com",
        "rahul@yahoo.com",
        "priya@outlook.com",
        "arjun@gmail.com"
    ],
    "Phone": [
        "9876543210",
        "9123456780",
        "9988776655",
        "9012345678"
    ],
    "City": [
        "Delhi",
        "Mumbai",
        "Bangalore",
        "Pune"
    ],
    "Department": [
        "Computer Science",
        "Information Technology",
        "Artificial Intelligence",
        "Software Engineering"
    ]
}

df = pd.DataFrame(data)

print("\nOriginal Structured Dataset:")
print(df.to_string(index=False))


# ------------------------------------------------------------
# Extract selected information
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("EXTRACTED INFORMATION")
print("-" * 70)

selected_data = df[
    ["Name", "Email", "Phone", "City"]
]

print(selected_data.to_string(index=False))


# ------------------------------------------------------------
# Filter records
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STUDENTS FROM DELHI")
print("-" * 70)

delhi_students = df[df["City"] == "Delhi"]

print(delhi_students.to_string(index=False))


# ============================================================
# PART 2: INFORMATION EXTRACTION FROM UNSTRUCTURED TEXT
# ============================================================

print("\n\n" + "=" * 70)
print("PART 2: INFORMATION EXTRACTION FROM UNSTRUCTURED TEXT")
print("=" * 70)


# Sample unstructured document
document_text = """
John Smith works at Microsoft Corporation in New York.
His email address is john.smith@microsoft.com and his phone number
is +1-987-654-3210.

He attended a technology conference on 15 March 2026.
The conference was organized by Microsoft and OpenAI.
John is currently working as a Software Engineer.

The company office is located in New York, United States.
"""


print("\nOriginal Unstructured Document:")
print(document_text)


# ------------------------------------------------------------
# 1. Named Entity Recognition
# ------------------------------------------------------------

doc = nlp(document_text)

print("\n" + "-" * 70)
print("NAMED ENTITIES EXTRACTED USING spaCy")
print("-" * 70)

for entity in doc.ents:

    print(
        f"Entity: {entity.text:30} "
        f"Type: {entity.label_:10} "
        f"Meaning: {spacy.explain(entity.label_)}"
    )


# ============================================================
# 2. Extract Email Addresses using Regular Expression
# ============================================================

print("\n" + "-" * 70)
print("EMAIL ADDRESSES")
print("-" * 70)

emails = re.findall(
    r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
    document_text
)

for email in emails:
    print(email)


# ============================================================
# 3. Extract Phone Numbers using Regular Expression
# ============================================================

print("\n" + "-" * 70)
print("PHONE NUMBERS")
print("-" * 70)

phone_pattern = r'\+?\d[\d\s().-]{8,}\d'

phones = re.findall(phone_pattern, document_text)

for phone in phones:
    print(phone.strip())


# ============================================================
# 4. Extract Dates using Regular Expression
# ============================================================

print("\n" + "-" * 70)
print("DATES")
print("-" * 70)

date_pattern = r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'

dates = re.findall(
    date_pattern,
    document_text
)

for date in dates:
    print(date)


# ============================================================
# 5. Extract Organizations, Persons and Locations
# ============================================================

print("\n" + "-" * 70)
print("IMPORTANT INFORMATION")
print("-" * 70)

persons = []
organizations = []
locations = []

for entity in doc.ents:

    if entity.label_ == "PERSON":
        persons.append(entity.text)

    elif entity.label_ == "ORG":
        organizations.append(entity.text)

    elif entity.label_ in ["GPE", "LOC"]:
        locations.append(entity.text)


print("Persons:")
for person in set(persons):
    print(" -", person)

print("\nOrganizations:")
for organization in set(organizations):
    print(" -", organization)

print("\nLocations:")
for location in set(locations):
    print(" -", location)


# ============================================================
# 6. Create Final Extracted Information
# ============================================================

print("\n" + "=" * 70)
print("FINAL INFORMATION EXTRACTION RESULT")
print("=" * 70)

print("Person       :", ", ".join(set(persons)))
print("Organization :", ", ".join(set(organizations)))
print("Location     :", ", ".join(set(locations)))
print("Email        :", ", ".join(emails))
print("Phone        :", ", ".join([p.strip() for p in phones]))
print("Date         :", ", ".join(dates))


# ============================================================
# 7. INFORMATION EXTRACTION PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("INFORMATION EXTRACTION PIPELINE")
print("=" * 70)

print("""
STRUCTURED DOCUMENT
        ↓
   Table / CSV
        ↓
   Pandas DataFrame
        ↓
 Select / Filter Data
        ↓
   Extract Information


UNSTRUCTURED DOCUMENT
        ↓
      Raw Text
        ↓
   Text Processing
        ↓
 ┌───────────────┬──────────────────┐
 │     spaCy     │   Regular        │
 │     NER       │   Expressions    │
 └───────────────┴──────────────────┘
        ↓
 Person / Organization / Location
 Email / Phone / Date
        ↓
   Extracted Information
""")
