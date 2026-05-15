import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    # Load pre-trained GloVe vectors (50-dim)
    # First run downloads; later runs load from cache
    print("Loading GloVe vectors (glove-wiki-gigaword-50)...")
    wv = api.load("glove-wiki-gigaword-50")

    # ----------------------------
    # Subtask A: Similarity
    # ----------------------------
    sim_robot_machine = wv.similarity("robot", "machine")
    print("\nA) Cosine similarity('robot', 'machine'):", sim_robot_machine)

    # ----------------------------
    # Subtask B: Vector arithmetic
    # King - Man + Woman = ?
    # ----------------------------
    result_b = wv.most_similar(positive=["king", "woman"], negative=["man"], topn=5)
    print("\nB) King - Man + Woman ≈")
    for word, score in result_b:
        print(f"   {word:10s} {score:.4f}")

    # ----------------------------
    # Subtask C: Analogy
    # "Kitchen" : "Chef" :: "Hospital" : ?
    # ----------------------------
    result_c = wv.most_similar(positive=["chef", "hospital"], negative=["kitchen"], topn=5)
    print("\nC) Kitchen : Chef :: Hospital : ?")
    for word, score in result_c:
        print(f"   {word:10s} {score:.4f}")

    # ----------------------------
    # PCA visualization
    # ----------------------------
    words = ['king', 'queen', 'man', 'woman', 'robot', 'ai', 'computer', 'pizza', 'burger', 'food']

    # Extract vectors (50 dims)
    vectors = []
    kept_words = []
    for w in words:
        if w in wv:
            vectors.append(wv[w])
            kept_words.append(w)
        else:
            print(f"Warning: '{w}' not in vocabulary, skipping.")

    # Reduce 50 -> 2 dimensions
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(vectors)

    # Plot
    plt.figure()
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1])

    # Label points
    for (x, y), w in zip(coords_2d, kept_words):
        plt.text(x, y, w)

    plt.title("GloVe Embeddings (50D) reduced to 2D with PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

if __name__ == "__main__":
    main()
