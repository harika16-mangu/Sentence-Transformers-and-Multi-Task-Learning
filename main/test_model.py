# train.py

import torch
from models.sentencetransformer import SentenceTransformer  # Correct import
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import numpy as np

def showcase_embeddings():
    sentences = [
        "I like watching tv shows",
        "I watch and play video games.",
        "Whiskers the cat chased the elusive red dot around the living room.",
        "The mountain stood tall and silent, shrouded in a blanket of fog.",
        "Quantum entanglement challenges our classical understanding of physics."
    ]
    
    # Initialize the Sentence Transformer model using the base model
    model = SentenceTransformer()
    embeddings = model.encode(sentences)
    
    # Compute cosine similarities
    print("Cosine Similarities:")
    num_sentences = len(sentences)
    cosine_sim_matrix = np.zeros((num_sentences, num_sentences))
    
    for i in range(num_sentences):
        for j in range(num_sentences):
            if i != j:
                sim = 1 - cosine(embeddings[i], embeddings[j])
                cosine_sim_matrix[i][j] = sim
                print(f"Cosine similarity between sentence {i+1} and {j+1}: {sim:.4f}")
            else:
                cosine_sim_matrix[i][j] = 1  # Similarity with itself

    # Display the cosine similarity matrix
    print("\nCosine Similarity Matrix:")
    print(cosine_sim_matrix)

    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=4)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Plot the embeddings
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(sentences):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y)
        plt.annotate(label, (x, y), fontsize=9, alpha=0.75)

    plt.title("t-SNE visualization of Sentence Embeddings")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.grid(True)
    plt.show()

def main():
    showcase_embeddings()

if __name__ == "__main__":
    main()
