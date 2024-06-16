# test_sentence_transformer_huggingface.py

from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def encode_sentences(model, sentences):
    # Encode sentences into embeddings
    embeddings = model.encode(sentences)
    return embeddings

def compute_cosine_similarity(embeddings):
    # Compute the cosine similarity matrix
    cosine_sim_matrix = util.cos_sim(embeddings, embeddings).numpy()
    return cosine_sim_matrix

def visualize_embeddings_tsne(embeddings, sentences):
    # Reduce dimensionality for visualization using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=2)
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

def display_cosine_similarity_matrix(cosine_sim_matrix, sentences):
    # Print the cosine similarity matrix with sentence labels
    print("Cosine Similarity Matrix:\n")
    print(" ", end=" ")
    for i in range(len(sentences)):
        print(f"{i:8}", end=" ")
    print()
    for i in range(len(cosine_sim_matrix)):
        print(f"{i} ", end=" ")
        for j in range(len(cosine_sim_matrix[i])):
            print(f"{cosine_sim_matrix[i][j]:.4f} ", end=" ")
        print()

def main():
    # Sample sentences
    sentences = [
        "Fetch Rewards offers a unique app-based loyalty program for grocery shoppers.",    
        "Fetch Rewards simplifies earning and redeeming rewards through everyday grocery purchases.",
        "The service was terrible and I am very disappointed.",
        "It's an average experience, nothing special.",
        "Quantum entanglement challenges our classical understanding of physics."
    ]

    # Load a pre-trained Sentence Transformer model from Hugging Face
    model = SentenceTransformer('bert-base-uncased')  # You can choose any model available on HF's Model Hub

    # Encode sentences into embeddings
    embeddings = encode_sentences(model, sentences)

    # Compute cosine similarity matrix
    cosine_sim_matrix = compute_cosine_similarity(embeddings)

    # Display the cosine similarity matrix
    display_cosine_similarity_matrix(cosine_sim_matrix, sentences)

    # Save embeddings (optional)
    np.save('embeddings_huggingface.npy', embeddings)

    # Visualize embeddings using t-SNE
    visualize_embeddings_tsne(embeddings, sentences)


if __name__ == "__main__":
    main()
