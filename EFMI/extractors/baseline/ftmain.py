import argparse
import torch
from torch.utils.data import DataLoader
from finetune import fine_tune
from models.resnet import get_adapted_resnet50
from ftextractor import AdaptedResnet50Extractor
from dataset.PatchedDataset import PatchedDataset, train_test_split_loaders
import classifier.svm as svm


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PatchedDataset(root_dir=args.root_dir, num_images=args.num_images)

    resnet50 = get_adapted_resnet50(latent_dim=args.latent_dim).cuda()

    train_loader, test_loader = train_test_split_loaders(
        dataset, args.batch_size, train_ratio=0.8
    )

    print(f"Finetuning {args.model_name}..")
    resnet50 = fine_tune(
        resnet50, train_loader, args.ft_epochs, args.model_name, device
    )

    extractor = AdaptedResnet50Extractor(model=resnet50, device=device)

    train_features, train_labels = extractor.extract_features(train_loader)
    test_features, test_labels = extractor.extract_features(test_loader)

    svm.classify_with_provided_splits(
        train_features,
        train_labels,
        test_features,
        test_labels,
        args.results_path,
        with_pca=True,
        pca_components=args.latent_dim,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract patches features with pre-trained nets"
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
        help="How may images to use (test purpose)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        help="Which pretrained baseline model to use as baseline feature extractor, defaults to resnet50. Availables: resnet50, densenet121",
    )
    parser.add_argument(
        "--ft_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Extracted latent vector dimension, defaults to 128",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Name of the experiment, save results in this path.",
    )
    args = parser.parse_args()

    main(args)
