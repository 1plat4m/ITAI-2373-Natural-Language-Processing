import re
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Helper: basic cleaner/tokenizer
# -------------------------
def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)      # keep letters/spaces only
    tokens = re.sub(r"\s+", " ", text).strip().split()
    return tokens

# -------------------------
# Helper: average GloVe word vectors
# -------------------------
def sentence_vector(text: str, wv):
    tokens = tokenize(text)
    vectors = [wv[t] for t in tokens if t in wv]

    if not vectors:
        # If none of the words exist in vocab, return a zero vector
        return np.zeros(wv.vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0)

def main():
    # Database of robot commands
    docs = ["Pick up the red ball", "Move to the kitchen", "Stop immediately", "Charge battery"]

    # User query
    query = "Grab the sphere"

    # Load GloVe vectors (50-dim)
    print("Loading GloVe vectors (glove-wiki-gigaword-50)...")
    wv = api.load("glove-wiki-gigaword-50")

    # Vectorize query and docs
    q_vec = sentence_vector(query, wv).reshape(1, -1)
    doc_vecs = np.vstack([sentence_vector(d, wv) for d in docs])

    # Cosine similarity: query vs each doc
    scores = cosine_similarity(q_vec, doc_vecs).flatten()

    # Pick best match
    best_idx = int(np.argmax(scores))
    best_doc = docs[best_idx]
    best_score = float(scores[best_idx])

    # Display results
    print("\nQuery:", query)
    print("\nSimilarity scores:")
    for d, s in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True):
        print(f"  {s:.4f}  ->  {d}")

    print(f"\nBest match: '{best_doc}'  (score={best_score:.4f})")

if __name__ == "__main__":
    main()
