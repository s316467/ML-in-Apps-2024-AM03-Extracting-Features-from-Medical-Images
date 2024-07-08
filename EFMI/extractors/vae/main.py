import argparse
import torch
from extractor import extract_latents
from dataset.PatchedDataset import PatchedDataset
from torch.utils.data import DataLoader
from EFMI.extractors.vae.train import train
import classifier.svm as svm


def main(args):
    """
    - Trains the VAE
    - Extract latents and labels
    - Classify latens with SVM
    """

    print("Training the VAE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PatchedDataset(root_dir=args.root_dir, num_images=args.num_images)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    VAE_trained = train(
        dataloader,
        device,
        latent_dim=args.latent_dim,
        num_epochs=args.num_epochs,
        vae_type=args.vae_type,
    )

    print("Extracting latents...")
    latents, labels = extract_latents(
        VAE_trained, dataloader, device, vae_type=args.vae_type
    )

    print("Classifying latents with SVMs...")
    svm.classify(latents, labels, args.results_path)


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
        default=32,
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--vae_type",
        type=str,
        default="vae",
        help="Choose which VAE to train, supported: [vae, resvae]",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Name of the experiment, save results in this path.",
    )

    args = parser.parse_args()

    main(args)
