import argparse
import torch
import numpy as np
from dataset.PatchedDataset import PatchedDataset
from torch.utils.data import DataLoader
from train import train
import classifier.svm as svm


# extract latent vectors for the entire dataset
def extract_latent_vectors(model, dataloader, device):
    latent_vectors = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, target, _, _ in dataloader:
            data = data.to(device)
            target = target.to(device)
            mu, logvar = model.encode(data)
            latent_vector = model.reparameterize(mu, logvar)
            latent_vectors.append(latent_vector.cpu().numpy())
            labels.append(target.cpu().numpy())
    return np.concatenate(latent_vectors), np.concatenate(labels)


def extract_latents(model, dataloader, device):
    latent_vectors, labels = extract_latent_vectors(model, dataloader, device)

    np.save("latent_vectors_vae.npy", latent_vectors)
    np.save("labels_vae.npy", labels)

    return (latent_vectors, labels)


def main(args):
    """
    - Trains the VAE
    - Extract latents and labels
    - Classify latens with SVM
    """

    print("Training the VAE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PatchedDataset(root_dir=args.root_dir, num_images=args.num_images)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    VAE_trained = train(dataloader, device, latent_dim=args.latent_dim)

    print("Extracting latents...")
    latents, labels = extract_latents(VAE_trained, dataloader, device)

    print("Classifying latents with SVMs...")
    svm.classify(latents, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a VAE with WSI patches, extrct latents and test with SVM"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to directory containing the patches folders",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    main(args)
