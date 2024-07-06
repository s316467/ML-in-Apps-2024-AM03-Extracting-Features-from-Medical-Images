import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Function to plot loss during training
def plot_training_loss(losses, output_dir):
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()


def plot_pca_variance(pca, output_dir):
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Variance Explained")
    plt.title("PCA - Variance Explained by Components")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "pca_variance.png"))
    plt.close()

    # Plot t-SNE of the PCA-reduced features and save to file


def plot_tsne(features, labels, title, filename):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    plt.figure(figsize=(10, 7))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis", s=5)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.savefig(filename)
    plt.close()

