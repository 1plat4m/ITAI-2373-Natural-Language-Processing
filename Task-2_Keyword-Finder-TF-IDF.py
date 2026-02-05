import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Corpus
documents = [
    "clean the kitchen",
    "clean the garage",
    "inventory the kitchen and the garage"
]

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the corpus
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert to DataFrame
df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=[f"Document {i+1}" for i in range(len(documents))]
)

print(df)
