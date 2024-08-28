import argparse
import torch
from extractor import extract_latents
from dataset.PatchedDataset import PatchedDataset, train_test_split_loaders
from train import train
import classifier.svm as svm
from models import res_vae


def main(args):
    """
    - Trains the VAE
    - Extract latents and labels
    - Classify latens with SVM
    """

    print(f"Training VAE {args.vae_type}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PatchedDataset(root_dir=args.root_dir, num_images=args.num_images)
    train_loader, test_loader = train_test_split_loaders(dataset, args.batch_size, 0.8)

    if args.no_train:
        print("Loading trained VAE...")
        VAE_trained = res_vae.ResVAE(args.latent_dim).to(device)
        VAE_trained.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    else:
        print("Starting VAE training...")
        VAE_trained = train(
            train_loader,
            device,
            latent_dim=args.latent_dim,
            num_epochs=args.num_epochs,
            vae_type=args.vae_type,
        )

    print("Extracting latents...")
    train_features, train_labels = extract_latents(
        VAE_trained, train_loader, device, args.results_path
    )
    test_features, test_labels = extract_latents(
        VAE_trained, test_loader, device, args.results_path
    )

    print("Classifying latents with SVMs...")
    svm.classify_with_provided_splits(
        train_features,
        train_labels,
        test_features,
        test_labels,
        args.results_path,
        with_pca=False,
    )


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
        default=128,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--vae_type",
        type=str,
        default="vae",
        help="Choose which VAE to train, supported: [vae, resvae]",
    )
    parser.add_argument(
        "--no_train",
        type=int,
        default=1,
        help="If specified vae is not trained, load the model from the specified path",
    )
    parser.add_argument("--model_path", type=str, help="path to trained model folder")
    parser.add_argument(
        "--results_path",
        type=str,
        help="Name of the experiment, save results in this path.",
    )

    args = parser.parse_args()
    args.no_train = True if args.no_train == 1 else False

    main(args)
