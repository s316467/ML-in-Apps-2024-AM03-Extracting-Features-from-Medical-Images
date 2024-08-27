import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Function to plot loss during training
def plot_training_loss(losses, output_dir, file_name):
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, file_name))
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
    plt.savefig(filename, dpi=500)
    plt.close()


def plot_losses(total_loss, mse, kld, beta):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(total_loss, label="Total Loss", color="blue")
    axs[0].set_title("Total Loss")
    axs[0].set_xlabel("steps")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(mse, label="MSE", color="green")
    axs[1].set_title("MSE")
    axs[1].set_xlabel("steps")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    axs[2].plot(kld, label="KLD", color="red")
    axs[2].set_title("kld")
    axs[2].set_xlabel("steps")
    axs[2].set_ylabel("Loss")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f"losses_vae_beta={beta}.png")
    # plt.show()
