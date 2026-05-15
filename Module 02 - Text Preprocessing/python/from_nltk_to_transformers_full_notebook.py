# %%
"""
Jupyter Notebook: From NLTK -> TF-IDF -> Word2Vec -> t-SNE/UMAP/PCA -> Transformer Embeddings
Features:
- Text cleaning with NLTK
- TF-IDF and top words
- Load GoogleNews Word2Vec (local file)
- 2D t-SNE and interactive 3D t-SNE (plotly)
- UMAP plots
- Transformer embeddings using SentenceTransformers (BERT-style)
- PCA vs t-SNE comparative visualization

Before running: install required packages (once):
!pip install nltk gensim sentence-transformers umap-learn plotly scikit-learn matplotlib

Place the GoogleNews Word2Vec binary at the path variable if you want to use it.
"""

# %%
# -- Imports & setup ---------------------------------------------------------
import os
import string
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import plotly.graph_objs as go
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer

# Jupyter friendly figures
plt.rcParams['figure.figsize'] = (10, 7)

# Download nltk resources if not present
nltk.download('punkt')
nltk.download('stopwords')

# %%
# -- Sample dataset and cleaning -------------------------------------------
raw_documents = [
    "Paris is the capital of France. It is known for the Eiffel Tower!",
    "Italy has Rome as its capital. The Colosseum is in Rome.",
    "Berlin is the capital of Germany and has a rich history.",
    "Madrid is the capital of Spain; it's famous for its art museums.",
    "Lisbon is a coastal city and the capital of Portugal."
]

stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    return " ".join(tokens)

cleaned_docs = [clean_text(d) for d in raw_documents]
print("Cleaned documents:\n", cleaned_docs)

# %%
# -- TF-IDF and top words --------------------------------------------------
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(cleaned_docs)
feature_names = vectorizer.get_feature_names_out()

def top_n_words_for_doc(tfidf_vector, feature_names, n=5):
    arr = tfidf_vector.toarray().flatten()
    idx = arr.argsort()[::-1][:n]
    return [(feature_names[i], arr[i]) for i in idx]

for i, doc in enumerate(cleaned_docs):
    print(f"\nTop words for doc {i}:")
    for w, s in top_n_words_for_doc(X_tfidf[i], feature_names, n=5):
        print(f"  {w}: {s:.4f}")

# %%
# -- Load Word2Vec (Google News) -------------------------------------------
# NOTE: Large binary (~1.5GB). Put the file in your working directory or change the path.
w2v_path = "GoogleNews-vectors-negative300.bin"
w2v = None
if os.path.exists(w2v_path):
    print("Loading Word2Vec model (this can take a minute)...")
    w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    print("Word2Vec loaded.")
else:
    print(f"Word2Vec binary not found at: {w2v_path} — skipping Word2Vec steps.")

# Example analogy if model loaded
if w2v is not None and all(w in w2v for w in ["Paris","France","Italy"]):
    print('\nAnalogy: Paris - France + Italy ->', w2v.most_similar(positive=["Paris","Italy"], negative=["France"], topn=5))

# %%
# -- Prepare vocabulary for embedding visualizations ------------------------
# Choose words of interest (capitals + countries + related cities)
words_of_interest = [
    "Paris","France","Rome","Italy","Berlin","Germany","Madrid","Spain",
    "Lisbon","Portugal","Eiffel","Colosseum","museum","capital"
]

# For Word2Vec embeddings (only if model loaded)
w2v_words = [w for w in words_of_interest if w in w2v] if w2v is not None else []

# %%
# -- Transformer embeddings (SentenceTransformers) --------------------------
# We'll embed the same words and also the whole documents for comparison
print("\nLoading SentenceTransformer (all-MiniLM-L6-v2)...")
st_model = SentenceTransformer('all-MiniLM-L6-v2')  # small and fast

# Embed words and documents
words_lower = [w.lower() for w in words_of_interest]
# sentence-transformers works best with phrases/sentences; for single words it's fine too.
word_embeddings = st_model.encode(words_lower, show_progress_bar=True)
doc_embeddings = st_model.encode(raw_documents, show_progress_bar=True)
print("Transformer embeddings done.")

# %%
# -- Function: run_dimensionality_reduction and plot (2D) -------------------

def plot_2d(points, labels, title="2D plot", annotate=True):
    plt.figure(figsize=(10,8))
    plt.scatter(points[:,0], points[:,1])
    if annotate:
        for i, lab in enumerate(labels):
            plt.annotate(lab, (points[i,0], points[i,1]))
    plt.title(title)
    plt.grid(True)
    plt.show()

# %%
# -- 2D t-SNE for transformer word embeddings -------------------------------
print('\nRunning 2D t-SNE on transformer word embeddings (this can take a while)...')
tsne_2d = TSNE(n_components=2, init='random', learning_rate='auto', perplexity=5, random_state=42)
word_tsne_2d = tsne_2d.fit_transform(word_embeddings)
plot_2d(word_tsne_2d, words_lower, title='2D t-SNE (Transformer word embeddings)')

# %%
# -- UMAP (2D) for transformer word embeddings ------------------------------
print('\nRunning UMAP (2D) on transformer word embeddings...')
umap_2d = umap.UMAP(n_components=2, random_state=42)
word_umap_2d = umap_2d.fit_transform(word_embeddings)
plot_2d(word_umap_2d, words_lower, title='2D UMAP (Transformer word embeddings)')

# %%
# -- PCA vs t-SNE comparison (Transformer doc embeddings) -------------------
print('\nPCA vs t-SNE on document embeddings...')
# PCA (2 components)
pca = PCA(n_components=2, random_state=42)
doc_pca = pca.fit_transform(doc_embeddings)

# t-SNE on same doc embeddings
tsne_docs = TSNE(n_components=2, init='random', learning_rate='auto', perplexity=5, random_state=42)
doc_tsne = tsne_docs.fit_transform(doc_embeddings)

# Plot side-by-side
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.scatter(doc_pca[:,0], doc_pca[:,1])
for i, txt in enumerate([f"doc_{i}" for i in range(len(raw_documents))]):
    plt.annotate(txt, (doc_pca[i,0], doc_pca[i,1]))
plt.title('PCA (documents)')
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(doc_tsne[:,0], doc_tsne[:,1])
for i, txt in enumerate([f"doc_{i}" for i in range(len(raw_documents))]):
    plt.annotate(txt, (doc_tsne[i,0], doc_tsne[i,1]))
plt.title('t-SNE (documents)')
plt.grid(True)
plt.show()

# %%
# -- Interactive 3D t-SNE (Plotly) for transformer word embeddings ---------
print('\nRunning 3D t-SNE (this may take a while for many points)...')
tsne_3d = TSNE(n_components=3, init='random', learning_rate='auto', perplexity=5, random_state=42)
word_tsne_3d = tsne_3d.fit_transform(word_embeddings)

trace = go.Scatter3d(
    x=word_tsne_3d[:,0],
    y=word_tsne_3d[:,1],
    z=word_tsne_3d[:,2],
    mode='markers+text',
    text=words_lower,
    textposition='top center',
    marker=dict(size=6)
)
layout = go.Layout(title='Interactive 3D t-SNE (Transformer word embeddings)')
fig = go.Figure(data=[trace], layout=layout)
fig.show()

# %%
# -- If Word2Vec is available: UMAP + t-SNE visualizations ------------------
if w2v is not None and len(w2v_words) > 0:
    print('\nVisualizing Word2Vec vocabulary subset...')
    w2v_embs = np.array([w2v[w] for w in w2v_words])

    # UMAP 2D
    umap2 = umap.UMAP(n_components=2, random_state=42)
    w2v_umap = umap2.fit_transform(w2v_embs)
    plot_2d(w2v_umap, w2v_words, title='Word2Vec: UMAP (2D)')

    # t-SNE 2D
    tsne_w2v = TSNE(n_components=2, init='random', learning_rate='auto', perplexity=5, random_state=42)
    w2v_tsne = tsne_w2v.fit_transform(w2v_embs)
    plot_2d(w2v_tsne, w2v_words, title='Word2Vec: t-SNE (2D)')

# %%
# -- Notes / Next steps ----------------------------------------------------
"""
- Tweak TSNE perplexity & learning_rate for better layouts with more data points
- For larger vocabularies use UMAP (faster) and sample intelligently
- Try interactive 3D UMAP by using the 3D option in UMAP and plotly
- For large transformer embeddings use PCA to reduce to ~50 dims before t-SNE
"""
